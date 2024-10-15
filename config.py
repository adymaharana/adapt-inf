import os

BASE_DATA_DIR = ''
BASE_IMG_DIR = ''

# Dataset filenames
LLAVA_RAW_DATASET = 'llava_v1_5_mix665k.json' # LLAVA
LLAVA_DATASET = 'llava_v1_5_mix665k_clean.json'  # LLAVA (multimodal samples with unavailable images are removed)
M3IT_DATASET = 'm3it_train.json' # M3IT
MINIGPT4_RAW_DATASET = 'filter_cap.json'
MINIGPT4_DATASET = 'minigpt4.json'
MANTIS_DATASET = 'mantis.json'
LAMM_DATASET = ''
VISIONFLAN_DATASET = ''

# Image Directory Paths
LLAVA_IMG_DIR = os.path.join(BASE_IMG_DIR, 'llava')
M3IT_IMG_DIR = os.path.join(BASE_IMG_DIR, 'm3it')
MINIGPT4_IMG_DIR = os.path.join(BASE_IMG_DIR, 'minigpt4')
MANTIS_IMG_DIR = os.path.join(BASE_IMG_DIR, 'mantis')

# Dataset paths
LLAVA_RAW_PATH = os.path.join(BASE_DATA_DIR, LLAVA_RAW_DATASET)
LLAVA_PATH = os.path.join(BASE_DATA_DIR, LLAVA_DATASET)
M3IT_PATH = os.path.join(BASE_DATA_DIR, M3IT_DATASET)
MINIGPT4_RAW_PATH = os.path.join(BASE_DATA_DIR, MINIGPT4_RAW_DATASET)
MINIGPT4_PATH = os.path.join(BASE_DATA_DIR, MINIGPT4_DATASET)
MANTIS_PATH = os.path.join(BASE_DATA_DIR, MANTIS_DATASET)

