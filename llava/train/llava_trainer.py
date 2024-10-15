import os
import time

from transformers.integrations import hp_params

import torch
import torch.nn as nn

from torch.utils.data import Sampler, SequentialSampler, RandomSampler

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)
from typing import List, Optional
import math, sys, shutil
from packaging import version
from transformers.trainer_utils import TrainOutput, speed_metrics
from transformers import is_torch_tpu_available #TODO: Is a replacement for is_torch_xla_available; will be deprecated in transformers 4.41.0
from transformers.training_args import ParallelMode
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.utils import is_sagemaker_mp_enabled, is_accelerate_available
from transformers import is_apex_available
from transformers.trainer_pt_utils import get_dataloader_sampler, get_model_param_count
from transformers.trainer_utils import HPSearchBackend
from transformers.trainer_callback import TrainerState
from transformers.integrations.deepspeed import deepspeed_load_checkpoint, deepspeed_init
from transformers.trainer import _is_peft_model
# from transformers.integrations.tpu import tpu_spmd_dataloader

TRAINER_STATE_NAME = "trainer_state.json"

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

else:
    IS_SAGEMAKER_MP_POST_1_10 = False

if is_apex_available():
    from apex import amp

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.spmd as xs
    import torch_xla.runtime as xr

if is_accelerate_available():
    from accelerate.utils import DistributedType
    from accelerate import skip_first_batches
    from accelerate import __version__ as accelerate_version
    DATA_SAMPLERS = [RandomSampler]
    if version.parse(accelerate_version) > version.parse("0.23.0"):
        from accelerate.data_loader import SeedableRandomSampler

        DATA_SAMPLERS += [SeedableRandomSampler]

import torch.distributed as dist


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        
        if self.args.sequential_sampler:
            return SequentialSampler(self.train_dataset)

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def save_optimizer(self):
        ''' Additional code to save optimizer at the end of the run '''
        output_dir = self._get_output_dir(trial=None)
        torch.save({'optimizer_state': self.optimizer.state_dict()}, os.path.join(output_dir, 'optimizer.pt'), _use_new_zipfile_serialization=False)

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)


    def _inner_training_loop(
            self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
        ):
            self.accelerator.free_memory()
            self._train_batch_size = batch_size
            if self.args.auto_find_batch_size:
                if self.state.train_batch_size != self._train_batch_size:
                    from accelerate.utils import release_memory

                    (self.model_wrapped,) = release_memory(self.model_wrapped)
                    self.model_wrapped = self.model

                    # Check for DeepSpeed *after* the intial pass and modify the config
                    if self.is_deepspeed_enabled:
                        # Temporarily unset `self.args.train_batch_size`
                        original_bs = self.args.per_device_train_batch_size
                        self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                        self.propagate_args_to_deepspeed(True)
                        self.args.per_device_train_batch_size = original_bs
                self.state.train_batch_size = self._train_batch_size
            logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
            # Data loader and number of training steps
            train_dataloader = self.get_train_dataloader()
            # if self.is_fsdp_xla_v2_enabled: # TODO: fix; v2 not in this version
            if self.is_fsdp_xla_enabled:
                pass # TODO: fix
                # train_dataloader = tpu_spmd_dataloader(train_dataloader)

            # Setting up training control variables:
            # number of training epochs: num_train_epochs
            # number of training steps per epoch: num_update_steps_per_epoch
            # total number of training steps to execute: max_steps
            total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

            len_dataloader = None
            num_train_tokens = None
            if has_length(train_dataloader):
                len_dataloader = len(train_dataloader)
                num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
                num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
                num_examples = self.num_examples(train_dataloader)
                if args.max_steps > 0:
                    max_steps = args.max_steps
                    num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                        args.max_steps % num_update_steps_per_epoch > 0
                    )
                    # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                    # the best we can do.
                    num_train_samples = args.max_steps * total_train_batch_size
                    if args.include_tokens_per_second:
                        num_train_tokens = (
                            self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
                        )
                else:
                    max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                    num_train_epochs = math.ceil(args.num_train_epochs)
                    num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
                    if args.include_tokens_per_second:
                        num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
            elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
                max_steps = args.max_steps
                # Setting a very large number of epochs so we go as many times as necessary over the iterator.
                num_train_epochs = sys.maxsize
                num_update_steps_per_epoch = max_steps
                num_examples = total_train_batch_size * args.max_steps
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
            else:
                raise ValueError(
                    "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                    f" {args.max_steps}"
                )

            if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
                if self.args.n_gpu > 1:
                    # nn.DataParallel(model) replicates the model, creating new variables and module
                    # references registered here no longer work on other gpus, breaking the module
                    raise ValueError(
                        "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                        " (torchrun or torch.distributed.launch (deprecated))."
                    )
                else:
                    debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

            delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

            # We need to reset the scheduler, as its parameters may be different on subsequent calls
            if self._created_lr_scheduler:
                self.lr_scheduler = None
                self._created_lr_scheduler = False

            if self.is_deepspeed_enabled:
                self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

            if not delay_optimizer_creation:
                self.create_optimizer_and_scheduler(num_training_steps=max_steps)

            self.state = TrainerState()
            self.state.is_hyper_param_search = trial is not None
            self.state.train_batch_size = self._train_batch_size

            # Compute absolute values for logging, eval, and save if given as ratio
            if args.logging_steps is not None:
                if args.logging_steps < 1:
                    self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
                else:
                    self.state.logging_steps = args.logging_steps
            if args.eval_steps is not None:
                if args.eval_steps < 1:
                    self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
                else:
                    self.state.eval_steps = args.eval_steps
            if args.save_steps is not None:
                if args.save_steps < 1:
                    self.state.save_steps = math.ceil(max_steps * args.save_steps)
                else:
                    self.state.save_steps = args.save_steps

            # Activate gradient checkpointing if needed
            if args.gradient_checkpointing:
                if args.gradient_checkpointing_kwargs is None:
                    gradient_checkpointing_kwargs = {}
                else:
                    gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs

                self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

            model = self._wrap_model(self.model_wrapped)

            # as the model is wrapped, don't use `accelerator.prepare`
            # this is for unhandled cases such as
            # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
            use_accelerator_prepare = True if model is self.model else False

            if delay_optimizer_creation:
                if use_accelerator_prepare:
                    self._fsdp_qlora_plugin_updates()
                    self.model = self.accelerator.prepare(self.model)
                self.create_optimizer_and_scheduler(num_training_steps=max_steps)

            # prepare using `accelerator` prepare
            if use_accelerator_prepare:
                self.model.train()
                if hasattr(self.lr_scheduler, "step"):
                    if self.use_apex:
                        model = self.accelerator.prepare(self.model)
                    else:
                        model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
                else:
                    # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                    model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                        self.model, self.optimizer, self.lr_scheduler
                    )

            if self.is_fsdp_enabled:
                self.model = self.model_wrapped = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

            # ckpt loading
            if resume_from_checkpoint is not None:
                if self.is_deepspeed_enabled:
                    deepspeed_load_checkpoint(
                        self.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(self.model)
                    )
                elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                    self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

            # Check if saved optimizer or scheduler states exist
            self._load_optimizer_and_scheduler(resume_from_checkpoint)

            # important: at this point:
            # self.model         is the Transformers Model
            # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
            # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

            # Train!
            logger.info("***** Running training *****")
            logger.info(f"  Num examples = {num_examples:,}")
            logger.info(f"  Num Epochs = {num_train_epochs:,}")
            logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
            if self.args.per_device_train_batch_size != self._train_batch_size:
                logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
            logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
            logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
            logger.info(f"  Total optimization steps = {max_steps:,}")
            logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

            self.state.epoch = 0
            start_time = time.time()
            epochs_trained = 0
            steps_trained_in_current_epoch = 0
            steps_trained_progress_bar = None

            # Check if continuing training from a checkpoint
            if resume_from_checkpoint is not None and os.path.isfile(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
            ):
                self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
                epochs_trained = self.state.global_step // num_update_steps_per_epoch
                if not args.ignore_data_skip:
                    steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                    steps_trained_in_current_epoch *= args.gradient_accumulation_steps
                else:
                    steps_trained_in_current_epoch = 0

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info(f"  Continuing training from epoch {epochs_trained}")
                logger.info(f"  Continuing training from global step {self.state.global_step}")
                if not args.ignore_data_skip:
                    logger.info(
                        f"  Will skip the first {epochs_trained} epochs then the first"
                        f" {steps_trained_in_current_epoch} batches in the first epoch."
                    )

            # Update the references
            self.callback_handler.model = self.model
            self.callback_handler.optimizer = self.optimizer
            self.callback_handler.lr_scheduler = self.lr_scheduler
            self.callback_handler.train_dataloader = train_dataloader
            if self.hp_name is not None and self._trial is not None:
                # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
                # parameter to Train when using DDP.
                self.state.trial_name = self.hp_name(self._trial)
            if trial is not None:
                assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
                self.state.trial_params = hp_params(assignments)
            else:
                self.state.trial_params = None
            # This should be the same if the state has been saved but in case the training arguments changed, it's safer
            # to set this after the load.
            self.state.max_steps = max_steps
            self.state.num_train_epochs = num_train_epochs
            self.state.is_local_process_zero = self.is_local_process_zero()
            self.state.is_world_process_zero = self.is_world_process_zero()

            # tr_loss is a tensor to avoid synchronization of TPUs through .item()
            tr_loss = torch.tensor(0.0).to(args.device)
            # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
            self._total_loss_scalar = 0.0
            self._globalstep_last_logged = self.state.global_step
            model.zero_grad()
            grad_norm: Optional[float] = None

            self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

            # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
            if not args.ignore_data_skip:
                for epoch in range(epochs_trained):
                    sampler = get_dataloader_sampler(train_dataloader)
                    sampler_kinds = [RandomSampler]
                    if version.parse(accelerate_version) > version.parse("0.23.0"):
                        sampler_kinds.append(SeedableRandomSampler)
                    is_random_sampler = isinstance(sampler, tuple(sampler_kinds))
                    if not is_random_sampler:
                        # We just need to begin an iteration to create the randomization of the sampler.
                        for _ in train_dataloader:
                            break
                    else:
                        # Otherwise we need to call the whooooole sampler cause there is some random operation added
                        # AT THE VERY END!
                        sampler = sampler if sampler is not None else []
                        _ = list(sampler)

            total_batched_samples = 0
            for epoch in range(epochs_trained, num_train_epochs):
                epoch_iterator = train_dataloader
                if hasattr(epoch_iterator, "set_epoch"):
                    epoch_iterator.set_epoch(epoch)

                # Reset the past mems state at the beginning of each epoch if necessary.
                if args.past_index >= 0:
                    self._past = None

                steps_in_epoch = (
                    len(epoch_iterator)
                    if len_dataloader is not None
                    else args.max_steps * args.gradient_accumulation_steps
                )
                self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

                if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                    self._load_rng_state(resume_from_checkpoint)

                rng_to_sync = False
                steps_skipped = 0
                if steps_trained_in_current_epoch > 0:
                    epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                    steps_skipped = steps_trained_in_current_epoch
                    steps_trained_in_current_epoch = 0
                    rng_to_sync = True

                step = -1
                for step, inputs in enumerate(epoch_iterator):
                    
                    total_batched_samples += 1

                    if self.args.include_num_input_tokens_seen:
                        main_input_name = getattr(self.model, "main_input_name", "input_ids")
                        if main_input_name not in inputs:
                            logger.warning(
                                "Tried to track the number of tokens seen, however the current model is "
                                "not configured properly to know what item is the input. To fix this, add "
                                "a `main_input_name` attribute to the model class you are using."
                            )
                        else:
                            self.state.num_input_tokens_seen += self.accelerator.gather(inputs[main_input_name]).numel()
                    if rng_to_sync:
                        self._load_rng_state(resume_from_checkpoint)
                        rng_to_sync = False

                    # Skip past any already trained steps if resuming training
                    if steps_trained_in_current_epoch > 0:
                        steps_trained_in_current_epoch -= 1
                        if steps_trained_progress_bar is not None:
                            steps_trained_progress_bar.update(1)
                        if steps_trained_in_current_epoch == 0:
                            self._load_rng_state(resume_from_checkpoint)
                        continue
                    elif steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.close()
                        steps_trained_progress_bar = None

                    if step % args.gradient_accumulation_steps == 0:
                        self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                    with self.accelerator.accumulate(model):
                        tr_loss_step = self.training_step(model, inputs)

                    if (
                        args.logging_nan_inf_filter
                        and not is_torch_tpu_available()
                        and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                    ):
                        # if loss is nan or inf simply add the average of previous logged losses
                        tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                    else:
                        if tr_loss.device != tr_loss_step.device:
                            raise ValueError(
                                f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                            )
                        tr_loss += tr_loss_step

                    self.current_flos += float(self.floating_point_ops(inputs))

                    is_last_step_and_steps_less_than_grad_acc = (
                        steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                    )

                    if (
                        total_batched_samples % args.gradient_accumulation_steps == 0
                        or
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        is_last_step_and_steps_less_than_grad_acc
                    ):
                        # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                        # in accelerate. So, explicitly enable sync gradients to True in that case.
                        if is_last_step_and_steps_less_than_grad_acc:
                            self.accelerator.gradient_state._set_sync_gradients(True)

                        # Gradient clipping
                        if args.max_grad_norm is not None and args.max_grad_norm > 0:
                            # deepspeed does its own clipping

                            if is_sagemaker_mp_enabled() and args.fp16:
                                _grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
                            elif self.use_apex:
                                # Revert to normal clipping otherwise, handling Apex or full precision
                                _grad_norm = nn.utils.clip_grad_norm_(
                                    amp.master_params(self.optimizer),
                                    args.max_grad_norm,
                                )
                            else:
                                _grad_norm = self.accelerator.clip_grad_norm_(
                                    model.parameters(),
                                    args.max_grad_norm,
                                )

                            if (
                                is_accelerate_available()
                                and self.accelerator.distributed_type == DistributedType.DEEPSPEED
                            ):
                                grad_norm = model.get_global_grad_norm()
                                # In some cases the grad norm may not return a float
                                if hasattr(grad_norm, "item"):
                                    grad_norm = grad_norm.item()
                            else:
                                grad_norm = _grad_norm

                        # Check for LoRA weights here
                        # total_A, total_B = 0, 0
                        # nonzero_A, nonzero_B = 0, 0
                        # for key, param in model.named_parameters():
                        #     if 'lora_A' in key:
                        #         total_A += param.shape[0]*param.shape[1]
                        #         nonzero_A += torch.count_nonzero(param)
                        #         if '30.self_attn.v_proj' in key:
                        #             print(key, param[0])
                        #     if 'lora_B' in key:
                        #         total_B += param.shape[0]*param.shape[1]
                        #         nonzero_B += torch.count_nonzero(param)
                        # print('Lora A: %s / %s = %.3f; Lora B: %s / %s = %.3f' % (nonzero_A, total_A, nonzero_A/total_A, nonzero_B, total_B, nonzero_B/total_B))

                        # Optimizer step
                        self.optimizer.step()
                        optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                        if optimizer_was_run:
                            # Delay optimizer scheduling until metrics are generated
                            if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                self.lr_scheduler.step()


                        model.zero_grad()
                        self.state.global_step += 1
                        self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                        self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                        # self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)
                        self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval) # grad norm is not sent in this version
                    else:
                        self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                    if self.control.should_epoch_stop or self.control.should_training_stop:
                        # PyTorch/XLA relies on the data loader to insert the mark_step for
                        # each step. Since we are breaking the loop early, we need to manually
                        # insert the mark_step here.
                        # if is_torch_xla_available():
                        if is_torch_tpu_available():
                            xm.mark_step()
                        break
                if step < 0:
                    logger.warning(
                        "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                        f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                        f" num_steps ({max_steps}) higher than the number of available samples."
                    )
                    self.control.should_training_stop = True

                self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
                # self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)
                self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

                if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                    # if is_torch_xla_available():
                    if is_torch_tpu_available():
                        # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                        xm.master_print(met.metrics_report())
                    else:
                        logger.warning(
                            "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                            "configured. Check your training configuration if this is unexpected."
                        )
                if self.control.should_training_stop:
                    break

            if args.past_index and hasattr(self, "_past"):
                # Clean the state at the end of training
                delattr(self, "_past")

            logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
            if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
                # Wait for everyone to get here so we are sure the model has been saved by process 0.
                # if is_torch_xla_available():
                if is_torch_tpu_available():
                    xm.rendezvous("load_best_model_at_end")
                elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                    dist.barrier()

                elif is_sagemaker_mp_enabled():
                    smp.barrier()

                self._load_best_model()

            # add remaining tr_loss
            self._total_loss_scalar += tr_loss.item()
            effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
            train_loss = self._total_loss_scalar / effective_global_step

            metrics = speed_metrics(
                "train",
                start_time,
                num_samples=num_train_samples,
                num_steps=self.state.max_steps,
                num_tokens=num_train_tokens,
            )
            self.store_flos()
            metrics["total_flos"] = self.state.total_flos
            metrics["train_loss"] = train_loss

            self.is_in_train = False

            self._memory_tracker.stop_and_update_metrics(metrics)

            self.log(metrics)

            run_dir = self._get_output_dir(trial)
            checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

            # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
            if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
                for checkpoint in checkpoints_sorted:
                    if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                        logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                        shutil.rmtree(checkpoint)

            self.control = self.callback_handler.on_train_end(args, self.state, self.control)

            # Wait for the checkpoint to be uploaded.
            self._finish_current_push()

            # After training we make sure to retrieve back the original forward pass method
            # for the embedding layer by removing the forward post hook.
            if self.neftune_noise_alpha is not None:
                self._deactivate_neftune(self.model)

            return TrainOutput(self.state.global_step, train_loss, metrics)
