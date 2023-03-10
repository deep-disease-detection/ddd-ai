from ddd.params import *
from ddd.ml_logic.models import *
from ddd.ml_logic.registry import save_model, save_result
import os
from ddd.ml_logic.registry import mlflow_run
import numpy as np

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
    },
    'cnn': {
        'init': initialize_CNN_model,
        'train': train_CNN_model
    }
}


def get_dataset():
    '''
    Return processed train, validation and test dataset from directory
    '''

    from tensorflow.keras.utils import image_dataset_from_directory

    print("Getting train dataset :)")
    #pour prendre un sample de train dataset
    if TEST == "True":
        train = image_dataset_from_directory(SAMPLE_PATH,
                                             labels='inferred',
                                             label_mode='categorical',
                                             shuffle=True,
                                             seed=42,
                                             color_mode="grayscale",
                                             batch_size=BATCH_SIZE)
    else:
        if AUGMENTED == "True":
            train = image_dataset_from_directory(AUGTRAIN_PATH,
                                                 labels='inferred',
                                                 label_mode='categorical',
                                                 shuffle=True,
                                                 seed=42,
                                                 color_mode="grayscale",
                                                 batch_size=BATCH_SIZE)
        else:
            train = image_dataset_from_directory(TRAIN_PATH,
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
def train_model(choice_model: str = 'custom'):

    #get all datasets
    train, val, test = get_dataset()

    model = MODEL_METHODS.get(choice_model).get('init')()

    model, history = MODEL_METHODS.get(choice_model).get('train')(model, train,
                                                                  val)

    params = {'model_type': choice_model, 'augmentation': AUGMENTED}

    metrics = {
        'history': history,
        'val_accuracy': np.max(history.history.get('val_accuracy')),
        'val_precision': np.max(history.history.get('val_precision')),
        'val_recall': np.max(history.history.get('val_recall'))
    }

    save_result(params=params, metrics=metrics)
    save_model(model)

    pass
