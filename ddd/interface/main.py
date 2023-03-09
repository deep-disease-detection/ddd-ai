from google.cloud import storage
from ddd.params import *
import os

def get_dataset():
    '''
    Return processed train, validation and test dataset from directory
    '''

    from tensorflow.keras.utils import image_dataset_from_directory

    print("Getting train dataset :)")
    train = image_dataset_from_directory(
        AUGTRAIN_PATH,
        labels='inferred',
        label_mode='categorical',
        image_size=IMAGE_SIZE
    )

    print('Getting validation dataset :)')
    validation = image_dataset_from_directory(
        VALIDATION_PATH,
        labels='inferred',
        label_mode='categorical',
        image_size= IMAGE_SIZE
    )

    print('Getting test dataset :)')
    test = image_dataset_from_directory(
        TEST_PATH,
        labels='inferred',
        label_mode='categorical',
        image_size= IMAGE_SIZE
    )

    print('All done !✅')
    return train, validation, test



def train_model(nom_model):
    '''
    Download processed dataset from local directory
    Train the model on the processed dataset
    Store training result
    '''

    train, val, test = get_dataset()
