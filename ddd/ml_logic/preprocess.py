# import std libraries
import os

# import data processing libraries
import numpy as np

# import images processing libraries
import cv2

# import sklearn classes and function
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

# import keras classes and functions
from tensorflow import keras
from keras.utils import to_categorical

# define global variables
path = "../data/TEM virus dataset/context_virus_1nm_256x256"
train_path = "augmented_train"
validation_path = "validation"
test_path = "test"


def extract_X_y_from_tif_image_folders(path):
    '''
    takes a path to a folder containing folders of images and returns
    a numpy array of images and a numpy array of labels
        Parameters:
            path (str): path to the folder containing the folders of images
        Returns:
            images (numpy array): numpy array of images
            y (numpy array): numpy array of labels
    '''

    images = []
    y = []
    for folder in os.listdir(path):
        if folder == '_EXCLUDED' or folder == 'crop_outlines':
            continue
        virus_type = folder.split('/')[-1]
        for file in os.listdir(os.path.join(path, folder)):
            img = cv2.imread(os.path.join(path, folder, file))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # extend the dimension of the image to be 3D
            img = np.expand_dims(img, axis=2)
            y.append(virus_type)
            images.append(img)
    return np.array(images), np.array(y)


def encoder_and_get_categories_from_y(y: np.ndarray):
    '''
    Encode the labels using LabelEncoder and convert them to categorical
        Parameters:
             y (np.ndarray): The labels of the data

        Returns:
            encoded_y (np.ndarray): The encoded labels

    '''
    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_y = encoder.transform(y)
    # convert the labels to categorical
    encoded_y = to_categorical(encoded_y)
    return encoded_y


def get_tif_images_from_directories(path: str, train_path: str, validation_path: str, test_path: str):
    '''
    takes a path to a folder containing folders of images and returns
    a numpy array of images and a numpy array of labels
        Parameters:
            path (str): path to the folder containing the folders of images
            train_path (str): path to the folder containing the train images
            validation_path (str): path to the folder containing the validation images
            test_path (str): path to the folder containing the test images
        Returns:
            X_train (numpy array): numpy array of train images
            y_train (numpy array): numpy array of train labels
            X_validation (numpy array): numpy array of validation images
            y_validation (numpy array): numpy array of validation labels
            X_test (numpy array): numpy array of test images
            y_test (numpy array): numpy array of test labels
    '''
    # get the train images and labels
    X_train, y_train = extract_X_y_from_tif_image_folders(os.path.join(path, train_path))
    y_train = encoder_and_get_categories_from_y(y_train)
    # get the validation images and labels
    X_validation, y_validation = extract_X_y_from_tif_image_folders(os.path.join(path, validation_path))
    y_validation = encoder_and_get_categories_from_y(y_validation)
    # get the test images and labels
    X_test, y_test = extract_X_y_from_tif_image_folders(os.path.join(path, test_path))
    y_test = encoder_and_get_categories_from_y(y_test)

    return X_train, y_train, X_validation, y_validation, X_test, y_test


def get_samples_of_data(X_train: np.array, y_train: np.array, X_validation: np.array, y_validation: np.array, X_test: np.array, y_test: np.array, sample_rate: float = 0.1):
    '''
    takes the original train, validation and test data and returns
    a sample of the data to be used for testing models and training
        Parameters:
            X_train (numpy array): numpy array of train images
            y_train (numpy array): numpy array of train labels
            X_validation (numpy array): numpy array of validation images
            y_validation (numpy array): numpy array of validation labels
            X_test (numpy array): numpy array of test images
            y_test (numpy array): numpy array of test labels
        Returns:
            X_train_samples (numpy array): numpy array of train images
            y_train_samples (numpy array): numpy array of train labels
            X_validation_samples (numpy array): numpy array of validation images
            y_validation_samples (numpy array): numpy array of validation labels
            X_test_samples (numpy array): numpy array of test images
            y_test_samples (numpy array): numpy array of test labels
    '''

    # shuffle the train data
    X_train, y_train = shuffle(X_train, y_train)
    # shuffle the validation data
    X_validation, y_validation = shuffle(X_validation, y_validation)
    # shuffle the test data
    X_test, y_test = shuffle(X_test, y_test)

    # get the number of samples to be used
    num_samples = int(X_train.shape[0] * sample_rate)
    X_train_samples = X_train[:num_samples]
    y_train_samples = y_train[:num_samples]
    X_validation_samples = X_validation[:num_samples]
    y_validation_samples = y_validation[:num_samples]
    X_test_samples = X_test[:num_samples]
    y_test_samples = y_test[:num_samples]

    return X_train_samples, y_train_samples, X_validation_samples, y_validation_samples, X_test_samples, y_test_samples
