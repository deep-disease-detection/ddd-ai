import os

IMAGE_SIZE = 256

GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCS_DATA_BUCKET = os.environ.get("GCS_DATA_BUCKET")

RAW_DATA_PATH = os.environ.get("RAW_DATA_PATH")
PROCESS_DATA_PATH = os.environ.get("PROCESS_DATA_PATH")
TO_PREPROCESS = os.environ.get("TO_PREPROCESS")

AUGTRAIN_PATH = os.environ.get("AUGTRAIN_PATH")
TRAIN_PATH = os.path.join(PROCESS_DATA_PATH, 'train')
VALIDATION_PATH = os.path.join(PROCESS_DATA_PATH, 'validation')
TEST_PATH = os.path.join(PROCESS_DATA_PATH, 'test')

VIRUSES = [
    'Adenovirus', 'Astrovirus', 'CCHF', 'Cowpox', 'Ebola', 'Influenza',
    'Lassa', 'Marburg', 'Nipah virus', 'Norovirus', 'Orf', 'Papilloma',
    'Rift Valley', 'Rotavirus'
]

IMAGES_PER_VIRUS = 736  #target number of imagettes for augmentation

#MODELING
METRICS = ['accuracy', 'recall', 'precision', 'f1']

BATCH_SIZE = 32
EPOCHS = 1
PATIENCE = 5

CHOICE_MODEL = 'dense'
