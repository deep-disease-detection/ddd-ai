from ddd.params import *
from ddd.ml_logic.models import *
import os
from ddd.ml_logic.registry import mlflow_run

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


def get_dataset(sample=False):
    '''
    Return processed train, validation and test dataset from directory
    '''

    from tensorflow.keras.utils import image_dataset_from_directory

    print("Getting train dataset :)")
    #pour prendre un sample de train dataset
    if sample==True:
        train = image_dataset_from_directory(SAMPLE_PATH,
                                         labels='inferred',
                                         label_mode='categorical',
                                         shuffle=True,
                                         seed=42,
                                         color_mode="grayscale",
                                         batch_size=BATCH_SIZE)
    else:
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



@mlflow_run
def train_model():

    #get all datasets
    train, val, test = get_dataset(TEST)

    model = MODEL_METHODS.get(CHOICE_MODEL).get('init')()

    model, history = MODEL_METHODS.get(CHOICE_MODEL).get('train')(model, train,
                                                                  val)

    print(f'Finished training the model')

    pass
