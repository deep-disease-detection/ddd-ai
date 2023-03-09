import os

IMAGE_SIZE = 256
MODEL_TARGET = os.environ.get('MODEL_TARGET')
TEST = os.environ.get('TEST')

GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCS_DATA_BUCKET = os.environ.get("GCS_DATA_BUCKET")

RAW_DATA_PATH = os.environ.get("RAW_DATA_PATH")
PROCESS_DATA_PATH = os.environ.get("PROCESS_DATA_PATH")
TO_PREPROCESS = os.environ.get("TO_PREPROCESS")

AUGTRAIN_PATH = os.environ.get("AUGTRAIN_PATH")
TRAIN_PATH = os.path.join(PROCESS_DATA_PATH, 'train')
VALIDATION_PATH = os.path.join(PROCESS_DATA_PATH, 'validation')
TEST_PATH = os.path.join(PROCESS_DATA_PATH, 'test')
SAMPLE_PATH = os.environ.get("SAMPLE_PATH")

VIRUSES = [
    'Adenovirus', 'Astrovirus', 'CCHF', 'Cowpox', 'Ebola', 'Influenza',
    'Lassa', 'Marburg', 'Nipah virus', 'Norovirus', 'Orf', 'Papilloma',
    'Rift Valley', 'Rotavirus'
]

IMAGES_PER_VIRUS = 736  #target number of imagettes for augmentation


LOCAL_REGISTRY_PATH= os.environ.get('LOCAL_REGISTRY_PATH')

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT")
MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME")


#MODELING
METRICS = ['accuracy', 'recall', 'precision', 'f1']

BATCH_SIZE = 1
EPOCHS = 1
PATIENCE = 5

CHOICE_MODEL = 'dense'
