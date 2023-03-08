import os

IMAGE_SIZE = 256

GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCS_DATA_BUCKET = os.environ.get("GCS_DATA_BUCKET")

RAW_DATA_PATH = os.environ.get("RAW_DATA_PATH")
PROCESS_DATA_PATH = os.environ.get("PROCESS_DATA_PATH")

VIRUSES = [
    'Adenovirus', 'Astrovirus', 'CCHF', 'Cowpox', 'Ebola', 'Influenza',
    'Lassa', 'Marburg', 'Nipah virus', 'Norovirus', 'Orf', 'Papilloma',
    'Rift Valley', 'Rotavirus'
]
