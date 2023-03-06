from google.cloud import storage
from ddd.params import *


def get_data():
    storage_client = storage.Client()

    # Note: Client.list_blobs requires at least package version 1.17.0.
    blobs = storage_client.list_blobs(GCS_DATA_BUCKET)

    # Note: The call returns a response only when the iterator is consumed.
    for blob in blobs:
        print(blob.name)


if __name__ == "__main__":
    get_data()
