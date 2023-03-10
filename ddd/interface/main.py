from ddd.params import *
from ddd.ml_logic.models import *
from ddd.ml_logic.registry import save_model, save_result
import os
from ddd.ml_logic.registry import mlflow_run
import numpy as np
from ddd.ml_logic.registry import load_model
from tensorflow.keras import Model
from ddd.ml_logic.preprocess import convert_b64_to_tf
import cv2
import base64

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
    'test':{
        'init': test_model,
        'train': train_test_model
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


# @mlflow_run
def train_model():

    #get all datasets
    train, val, test = get_dataset()

    model = MODEL_METHODS.get(CHOICE_MODEL).get('init')()

    model, history = MODEL_METHODS.get(CHOICE_MODEL).get('train')(model, train,
                                                                  val)

    val_accuracy = np.min(history.history.get('val_accuracy'))

    params = {'model_type': CHOICE_MODEL}

    save_result(params=params, metrics=dict(acc=val_accuracy))
    save_model(model)

    return val_accuracy



def evaluate_model() -> float:
    """
    Evaluate the performance of the latest production model on processed data
    Return ...
    """
    train, val, test = get_dataset()
    model = load_model()
    assert model is not None

    metrics_dict = model.evaluate(
        test,
        batch_size = BATCH_SIZE,
        verbose= 0,
        return_dict = True
    )

    loss = metrics_dict['loss']
    accuracy = metrics_dict['accuracy']

    params = dict(
        context='evaluate'
    )

    print("The model loss is ", loss)
    print("The model accuracy is ", accuracy)

    save_result(params=params, metrics=metrics_dict)
    return accuracy



def predict(image:tf):

    model = load_model()
    assert model is not None
    image = np.expand_dims(image, axis=0)    #pour avoir le bon format
    y_pred = model.predict(image)
    max = y_pred.argmax()
    label = CLASS_NAME[max]
    print(label)
    print(y_pred[0][max])
    proba = y_pred[0][max]
    return label, proba


if __name__ == '__main__':
    image = cv2.imread('data/process/TEM virus dataset/context_virus_1nm_256x256/augmented_train/Adenovirus/A4-65k-071120_2_0.png')
    image2 = cv2.imencode('.png',image)[1]
    b64 =base64.b64encode(image2)
    X = convert_b64_to_tf(b64)
    predict(X)
