from google.cloud import storage
from ddd.params import *
import os

def get_dataset(**kwargs):
    '''
    Return processed train, validation and test dataset from directory
    '''

    from tensorflow.keras.utils import image_dataset_from_directory

    print("Getting train dataset :)")
    train = image_dataset_from_directory(
        AUGTRAIN_PATH,
        labels='inferred',
        label_mode='categorical',
        shuffle=True,
        seed=42,
        color_mode="grayscale",
        batch_size= kwargs.get('batch_size', 32)
    )

    print('Getting validation dataset :)')
    validation = image_dataset_from_directory(
        VALIDATION_PATH,
        labels='inferred',
        label_mode='categorical',
        shuffle=True,
        seed=42,
        color_mode="grayscale",
        batch_size= kwargs.get('batch_size', 32)
    )

    print('Getting test dataset :)')
    test = image_dataset_from_directory(
        TEST_PATH,
        labels='inferred',
        label_mode='categorical',
        shuffle=True,
        seed=42,
        color_mode="grayscale",
        batch_size= kwargs.get('batch_size', 32)
    )

    print('All done !✅')
    return train, validation, test
