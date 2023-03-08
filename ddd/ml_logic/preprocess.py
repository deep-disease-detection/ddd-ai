# import std libraries
import os
import csv
import sys

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
import tensorflow as tf

# define global variables
path = "../data/TEM virus dataset/context_virus_1nm_256x256"
train_path = "augmented_train"
validation_path = "validation"
test_path = "test"

# import classes and functions from ddd
from ddd.params import *
from ddd.utils import average_coord


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


def get_dic_center(split_set: str = 'train'):
    '''
        Create a dictionary with all coordinates center for each virus and each file
    '''
    #dictionary initiation
    train_data_dic = {}
    list_virus = [
        v for v in os.listdir(os.path.join(RAW_DATA_PATH, split_set))
        if v[0] != '.'
    ]
    for virus in list_virus:
        train_data_dic[virus] = {}

    #filling the dictinary virus by virus
    for virus in list_virus:
        path_position_file = os.path.join(RAW_DATA_PATH, split_set, virus,
                                          'particle_positions')
        list_position_file = [
            f for f in os.listdir(path_position_file) if f[0] != '.'
        ]
        #looping over file
        for file in list_position_file:
            with open(
                    os.path.join(RAW_DATA_PATH, split_set, virus,
                                 'particle_positions', file), 'r') as f:
                lines = f.readlines()
                center_coords = []
                particle = []
                for i in range(3, len(lines)):
                    if lines[i] != 'particleposition\n':
                        coordinate = tuple(
                            float(c) for c in lines[i].strip('\n').split(';'))
                        particle.append(coordinate)
                        if i == len(lines) - 1 and len(particle) == 2:
                            center_coords.append(
                                average_coord(particle[0], particle[1]))
                        elif i == len(lines) - 1:
                            for coord in particle:
                                center_coords.append(coord)
                    else:
                        #compute the center between 2 center point of one single particule
                        if len(particle) == 2:
                            center_coords.append(
                                average_coord(particle[0], particle[1]))
                            particle = []
                        else:
                            for coord in particle:
                                center_coords.append(coord)
                            particle = []
                #file_name = file.rstrip('_particlepositions.txt')
                file_name = file.replace('_particlepositions.txt', '')
                train_data_dic[virus][file_name] = center_coords

    return train_data_dic


def get_pic_mesure(split_set: str = 'train'):
    '''
        return a dictionary with image mesurement and scale
    '''

    pic_data_dic = {}
    list_virus = [
        v for v in os.listdir(os.path.join(RAW_DATA_PATH, split_set))
        if v[0] != '.'
    ]

    for virus in list_virus:
        pic_data_dic[virus] = {}

    for virus in list_virus:
        path_tag_files = os.path.join(RAW_DATA_PATH, split_set, virus, 'tags')
        list_tag_files = os.listdir(path_tag_files)
        for file in list_tag_files:
            with open(f'{path_tag_files}/{file}', 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter=';')
                all_infos = []
                for row in reader:
                    all_infos.append(row)
                #file_name = file.rstrip('.tif_tags.csv')
                file_name = file.replace('.tif_tags.csv', '')
                pic_data_dic[virus][file_name] = {}
                pic_data_dic[virus][file_name]['Height'] = int(all_infos[0][1])
                pic_data_dic[virus][file_name]['Width'] = int(all_infos[1][1])
                pic_data_dic[virus][file_name]['Xscale'] = float(
                    all_infos[5][1].replace(',', '.'))
                pic_data_dic[virus][file_name]['Yscale'] = float(
                    all_infos[6][1].replace(',', '.'))
    return pic_data_dic


def resize_image(img: np.array, file: str, pic_dict: dict) -> np.array:
    '''
    Method that resizes an image to obtain a resolution of 1px = 1nm
    Uses picture meta data from pic_dict
    Uses tf.image.resize with Lanczos3 kernel interpolation
    '''

    new_width = round(pic_dict[file]['Height'] * pic_dict[file]['Yscale'])
    new_height = round(pic_dict[file]['Width'] * pic_dict[file]['Xscale'])

    resized_img = tf.image.resize(img,
                                  size=(new_width, new_height),
                                  method=tf.image.ResizeMethod.LANCZOS3)
    return resized_img.numpy()


def resize_particles(particles: list, file: str, pic_dict: dict) -> list:
    '''
    Method that repositions the particles following the resize
    Uses picture meta data from pic_dict
    '''
    for i in range(len(particles)):
        particles[i] = (particles[i][0] * pic_dict[file]['Yscale'],
                        particles[i][1] * pic_dict[file]['Xscale'])
    return particles


def make_imagettes(img: np.array, particles: list) -> list:
    imagettes = []

    for p in particles:
        offset_height = round(p[1] + IMAGE_SIZE - IMAGE_SIZE / 2)
        offset_width = round(p[0] + IMAGE_SIZE - IMAGE_SIZE / 2)
        imagettes.append(
            tf.image.crop_to_bounding_box(img,
                                          offset_height=offset_height,
                                          offset_width=offset_width,
                                          target_height=IMAGE_SIZE,
                                          target_width=IMAGE_SIZE).numpy())

    return imagettes


def save_imagettes(imagettes: list, split_set: str, virus: str, file: str,
                   format: str):
    for i, img in enumerate(imagettes):
        imagette_file_name = f'{file}_{str(i)}.{format}'
        destination_path = os.path.join(PROCESS_DATA_PATH, split_set, virus,
                                        imagette_file_name)
        cv2.imwrite(destination_path, img)


def preprocess_viruses(split_set: str = 'train'):

    #Check if process directories exist, otherwise create them
    try:
        os.listdir(PROCESS_DATA_PATH)
    except FileNotFoundError:
        print("There is no process directory, creating it for you")
        os.makedirs(os.path.join(PROCESS_DATA_PATH, split_set))

    #Loading structure and data dictionnaries
    particle_dict = get_dic_center()
    pic_dict = get_pic_mesure()

    #Subsetting to specific folders if test mode
    print(f'Executing in test mode for {VIRUSES}')
    particle_dict = {k: particle_dict.get(k) for k in VIRUSES}
    pic_dict = {k: pic_dict.get(k) for k in VIRUSES}

    vcount = 0
    for virus, files in particle_dict.items():
        #Printing progress
        vcount = vcount + 1
        print(f"Preprocessing virus folder {vcount} of {len(particle_dict)} 🦠")

        #Check if directory for processed virus data already exists, otherwise make it
        if virus in os.listdir(os.path.join(PROCESS_DATA_PATH, split_set)):
            print(f"Skipping over {virus} because it was already processed")
            continue
        else:
            print(f"Creating directory for {virus}")
            os.mkdir(os.path.join(PROCESS_DATA_PATH, split_set, virus))

        image_count = 0
        for file, particles in files.items():
            image_count = image_count + 1

            print(
                f"-------Preprocessing image file {image_count} of {len(files)} 🎆"
            )

            #Load the image
            image_path = os.path.join(RAW_DATA_PATH, split_set, virus,
                                      f'{file}.tif')

            img = cv2.imread(image_path, -1).astype(np.float32)

            if len(img.shape) == 2:
                img = np.expand_dims(
                    cv2.imread(image_path, -1).astype(np.float32), 2)
            elif len(img.shape) > 2:
                img = np.expand_dims(img[:, :, 1], 2)

            #Resize the image
            img = resize_image(img, file, pic_dict[virus])

            #Apply the same transformation to the particles
            resized_particles = resize_particles(particles, file,
                                                 pic_dict[virus])

            #Min-max scale the image
            img = (cv2.normalize(img, None, 1.0, 0.0, cv2.NORM_MINMAX) *
                   255).astype(np.uint8)

            #Add padding to the image
            img = cv2.copyMakeBorder(img, IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE,
                                     IMAGE_SIZE, cv2.BORDER_REFLECT)

            img = np.expand_dims(img,
                                 2)  #reshape to (H, W, 1) for tensforflow :)

            #Crop around each particle
            imagettes = make_imagettes(img, resized_particles)

            #Create the files
            save_imagettes(imagettes, split_set, virus, file, "png")


if __name__ == "__main__":
    preprocess_viruses(sys.argv[1])
