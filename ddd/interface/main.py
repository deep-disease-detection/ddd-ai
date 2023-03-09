from ddd.params import *
from ddd.ml_logic.models import *
import os

MODEL_METHODS = {
    'custom': {
        'init': initialize_custom_model,
        'train': train_custom_model_fromdataset
    },
    'dense': {
        'init': initialize_DenseNet_model,
        'train': train_DenseNet_model_fromdataset
    },
    'vgg19': {
        'init': VGG19_model,
        'train': train_VGG19_model_fromdataset
    }
}


def get_dataset():
    '''
    Return processed train, validation and test dataset from directory
    '''

    from tensorflow.keras.utils import image_dataset_from_directory

    print("Getting train dataset :)")
    train = image_dataset_from_directory(AUGTRAIN_PATH,
                                         labels='inferred',
                                         label_mode='categorical',
                                         shuffle=True,
                                         seed=42,
                                         color_mode="grayscale",
                                         batch_size=BATCH_SIZE)

    print('Getting validation dataset :)')
    validation = image_dataset_from_directory(VALIDATION_PATH,
                                              labels='inferred',
                                              label_mode='categorical',
                                              shuffle=True,
                                              seed=42,
                                              color_mode="grayscale",
                                              batch_size=BATCH_SIZE)

    print('Getting test dataset :)')
    test = image_dataset_from_directory(TEST_PATH,
                                        labels='inferred',
                                        label_mode='categorical',
                                        shuffle=True,
                                        seed=42,
                                        color_mode="grayscale",
                                        batch_size=BATCH_SIZE)

    print('All done !✅')
    return train, validation, test


def train_model(model_choice: str):

    #get all datasets
    train, val, test = get_dataset()

    model = MODEL_METHODS.get(model_choice).get('init')()

    model, history = MODEL_METHODS.get(model_choice).get('train')(model, train,
                                                                  val)

    print(f'Finished training the model')

    pass
