# import std libraries
import os
import csv
import sys
from itertools import product

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
from shutil import copyfile

#pour la fonction convert base64
from imageio import imread
import base64
import io

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


def get_tif_images_from_directories(path: str, train_path: str,
                                    validation_path: str, test_path: str):
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
    X_train, y_train = extract_X_y_from_tif_image_folders(
        os.path.join(path, train_path))
    y_train = encoder_and_get_categories_from_y(y_train)
    # get the validation images and labels
    X_validation, y_validation = extract_X_y_from_tif_image_folders(
        os.path.join(path, validation_path))
    y_validation = encoder_and_get_categories_from_y(y_validation)
    # get the test images and labels
    X_test, y_test = extract_X_y_from_tif_image_folders(
        os.path.join(path, test_path))
    y_test = encoder_and_get_categories_from_y(y_test)

    return X_train, y_train, X_validation, y_validation, X_test, y_test


def get_samples_of_data(X_train: np.array,
                        y_train: np.array,
                        X_validation: np.array,
                        y_validation: np.array,
                        X_test: np.array,
                        y_test: np.array,
                        sample_rate: float = 0.1):
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


def rotate_flip(img_path, nbr_passage: int):

    degree = [
        cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE
    ]
    flipping_param = [1, 0, -1]
    combinaison = list(product(degree, flipping_param))
    img = cv2.imread(img_path)
    new_image = cv2.rotate(img, combinaison[nbr_passage][0])
    new_image = cv2.flip(new_image, combinaison[nbr_passage][1])
    return new_image


def save_imagettes(imagettes: list, split_set: str, virus: str, file: str,
                   format: str):
    for i, img in enumerate(imagettes):
        imagette_file_name = f'{file}_{str(i)}.{format}'
        destination_path = os.path.join(PROCESS_DATA_PATH, split_set, virus,
                                        imagette_file_name)
        cv2.imwrite(destination_path, img)


def preprocess_viruses():

    #Check if process directories exist, otherwise create them
    try:
        os.listdir(os.path.join(PROCESS_DATA_PATH, TO_PREPROCESS))
    except FileNotFoundError:
        print("There is no process directory, creating it for you")
        os.makedirs(os.path.join(PROCESS_DATA_PATH, TO_PREPROCESS))

    #Loading structure and data dictionnaries
    particle_dict = get_dic_center(TO_PREPROCESS)
    pic_dict = get_pic_mesure(TO_PREPROCESS)

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
        if virus in os.listdir(os.path.join(PROCESS_DATA_PATH, TO_PREPROCESS)):
            print(f"Skipping over {virus} because it was already processed")
            continue
        else:
            print(f"Creating directory for {virus}")
            os.mkdir(os.path.join(PROCESS_DATA_PATH, TO_PREPROCESS, virus))

        image_count = 0
        for file, particles in files.items():
            image_count = image_count + 1

            print(
                f"-------Preprocessing image file {image_count} of {len(files)} 🎆"
            )

            #Load the image
            image_path = os.path.join(RAW_DATA_PATH, TO_PREPROCESS, virus,
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
            save_imagettes(imagettes, TO_PREPROCESS, virus, file, "png")


def augment_pictures():

    virus_list = os.listdir(TRAIN_PATH)

    print('Starting image augmentation script 🦠')

    #Check if process directories exist, otherwise create them
    try:
        os.listdir(AUGTRAIN_PATH)
    except FileNotFoundError:
        print("There is no augmented train directory, creating it")
        os.makedirs(AUGTRAIN_PATH)

    for virus in virus_list:
        source_path = os.path.join(TRAIN_PATH, virus)
        dest_virus_folder = os.path.join(AUGTRAIN_PATH, virus)
        file_name_list = os.listdir(source_path)

        if virus in os.listdir(AUGTRAIN_PATH):
            pass
        else:
            print(f"Creating directory for {virus}")
            os.mkdir(os.path.join(AUGTRAIN_PATH, virus))

        nbr_of_imagettes = len(os.listdir(source_path))
        if nbr_of_imagettes < IMAGES_PER_VIRUS:
            # copy paste all images in another directory
            for image in file_name_list:
                copyfile(os.path.join(source_path, image),
                         os.path.join(dest_virus_folder, image))

            #augment the imagettes
            nbre_passage = 0
            while len(os.listdir(dest_virus_folder)) < IMAGES_PER_VIRUS:
                for image in file_name_list:
                    new_image = rotate_flip(os.path.join(source_path, image),
                                            nbre_passage)
                    print(f'AUG{nbre_passage}{image}')
                    print(len(os.listdir(dest_virus_folder)))
                    cv2.imwrite(
                        os.path.join(dest_virus_folder,
                                     f'AUG_{nbre_passage}_{image}'), new_image)
                    if len(os.listdir(dest_virus_folder)) == IMAGES_PER_VIRUS:
                        print('done')
                        break
                nbre_passage = nbre_passage + 1
                if nbre_passage >= 9:
                    print(
                        'not enough pictures to reach target imagette number')

        else:
            # randomly copy images
            for _ in range(IMAGES_PER_VIRUS):
                index = np.random.randint(len(file_name_list))
                file_name = file_name_list[index]

                #pour ne pas faire de doublons
                if file_name not in os.listdir(
                        os.path.join(AUGTRAIN_PATH, virus)):
                    copyfile(os.path.join(source_path, file_name),
                             os.path.join(AUGTRAIN_PATH, virus, file_name))

                file_name_list.pop(index)


def make_sample():
    '''faire un petit dataset pour les test'''

    for virus in os.listdir(AUGTRAIN_PATH):
        if virus in os.listdir(SAMPLE_PATH):
            pass
        else:
            print(f"Creating directory for {virus}")
            print(SAMPLE_PATH)
            os.mkdir(os.path.join(SAMPLE_PATH, virus))
        for i in range(10):
            virus_path = os.path.join(AUGTRAIN_PATH, virus)
            file_name = os.listdir(virus_path)[i]

            copyfile(os.path.join(virus_path, file_name),
                     os.path.join(SAMPLE_PATH, virus, file_name))


def convert_b64_to_tf(b64code):
    '''convert the base64 code of the image to a tensorflow shape (256,256, 1)'''
    b = base64.b64decode(b64code.decode('utf-8'))
    im = imread(io.BytesIO(b))
    im2 = im[:, :, 1]
    im2 = np.expand_dims(im2, axis=2)
    return tf.constant(im2)


def get_dic_center_yolo(split_set: str = 'train'):
    '''
        Create a dictionary with all coordinates center for each virus and each file
    '''
    #dictionary initiation
    train_data_dic = {}
    list_virus = [
        v for v in os.listdir(os.path.join(RAW_DATA_PATH, split_set))
        if v[0] != '.'
    ]

    #filling the dictinary virus by virus
    for virus in list_virus:
        train_data_dic[virus] = {}
        path_position_file = os.path.join(RAW_DATA_PATH, split_set, virus,
                                          'particle_positions')

        #getting all anotation files
        list_position_file = [
            f for f in os.listdir(path_position_file) if f[0] != '.'
        ]
        #looping over each annotation file
        for file in list_position_file:
            with open(
                    os.path.join(RAW_DATA_PATH, split_set, virus,
                                 'particle_positions', file), 'r') as f:
                lines = f.readlines()
                particles = []
                particle = []
                for i in range(3, len(lines)):
                    if lines[i] != 'particleposition\n':
                        coordinate = tuple(
                            float(c) for c in lines[i].strip('\n').split(';'))
                        particle.append(coordinate)

                        #handle last particle of file
                        if i == len(lines) - 1 and len(particle) == 2:
                            particles.append(
                                [average_coord(particle[0], particle[1])])
                        elif i == len(lines) - 1:
                            particles.append(particle)
                    else:
                        #compute the center between 2 center point of one single particle
                        if len(particle) == 2:
                            particles.append(
                                [average_coord(particle[0], particle[1])])
                            particle = []
                        else:
                            particles.append(particle)
                            particle = []
                file_name = file.replace('_particlepositions.txt', '')
                train_data_dic[virus][file_name] = particles

    return train_data_dic


def resize_particles_yolo(particles: list, file: str, pic_dict: dict):
    return [[(c[0] * pic_dict[file]['Yscale'], c[1] * pic_dict[file]['Xscale'])
             for c in particle] for particle in particles]


def check_box_in_pic(x, y, w, h):
    return x - w / 2 >= 0 and x + w / 2 <= YOLO_IMAGE_SIZE and y - h / 2 >= 0 and y + h / 2 <= YOLO_IMAGE_SIZE


def preprocess_yolo(split_set: str = 'train'):

    particle_dict = get_dic_center_yolo(split_set)
    pic_dict = get_pic_mesure(split_set)

    for virus in VIRUS_METADATA.keys():

        print(f"Preparing {virus} for YOLOv7 🦠")

        image_file_count = len(pic_dict.get(virus))
        for i, image_file_name in enumerate(pic_dict.get(virus)):

            print(f"--------- Processing image {i+1}/{image_file_count} 🎞️")

            image_path = os.path.join(RAW_DATA_PATH, split_set, virus,
                                      f'{image_file_name}.tif')

            img = cv2.imread(image_path, -1).astype(np.float32)

            #handle images that have 3 channels
            if len(img.shape) == 2:
                img = np.expand_dims(
                    cv2.imread(image_path, -1).astype(np.float32), 2)
            elif len(img.shape) > 2:
                img = np.expand_dims(img[:, :, 1], 2)

            #get particles
            particles = particle_dict.get(virus).get(image_file_name)

            #resize image and particles
            img = resize_image(img, image_file_name, pic_dict.get(virus))
            r_particles = resize_particles_yolo(particles, image_file_name,
                                                pic_dict.get(virus))

            #crop images to 1000x1000 pixels
            x_coordinates = []
            y_coordinates = []
            for particle in r_particles:
                x_coordinates += [p[0] for p in particle]
                y_coordinates += [p[1] for p in particle]

            #compute mean of all particles
            central_particle = (int(np.mean(x_coordinates)),
                                int(np.mean(y_coordinates)))

            #compute the margin we need to pad if necessary
            top = central_particle[1] + YOLO_IMAGE_SIZE / 2
            bottom = central_particle[1] - YOLO_IMAGE_SIZE / 2
            right = central_particle[0] + YOLO_IMAGE_SIZE / 2
            left = central_particle[0] - YOLO_IMAGE_SIZE / 2

            border_top = int(max(top - img.shape[1], 0))
            border_bottom = int(abs(min(bottom, 0)))
            border_right = int(max(right - img.shape[0], 0))
            border_left = int(abs(min(left, 0)))

            #pad if necessary
            crop_img = cv2.copyMakeBorder(img,
                                          border_bottom,
                                          border_top,
                                          border_left,
                                          border_right,
                                          borderType=cv2.BORDER_CONSTANT)

            #crop
            crop_img = crop_img[int(bottom + border_bottom):int(top +
                                                                border_bottom),
                                int(left + border_left):int(right +
                                                            border_left), ]

            #adapt particle positions to crop
            final_particles = [[(p[0] - left, p[1] - bottom) for p in particle]
                               for particle in r_particles]

            #make the bounding boxes
            image_labels = []

            for f_particle in final_particles:
                #pick a color for the particle

                #if there is only one coordinate, we make a box using virus metadata
                if len(f_particle) == 1:

                    #if only 1 coordinate set and elongated, simply skip
                    if VIRUS_METADATA.get(virus).get("elongated"):
                        continue

                    yolo_x = f_particle[0][0]
                    yolo_y = f_particle[0][1]
                    yolo_w = VIRUS_METADATA.get(virus)['diameter']
                    yolo_h = VIRUS_METADATA.get(virus)['diameter']

                    if not check_box_in_pic(yolo_x, yolo_y, yolo_w, yolo_h):
                        continue

                    #Normalize x,y, w and h
                    yolo_x = yolo_x / YOLO_IMAGE_SIZE
                    yolo_y = yolo_y / YOLO_IMAGE_SIZE
                    yolo_h = yolo_h / YOLO_IMAGE_SIZE
                    yolo_w = yolo_w / YOLO_IMAGE_SIZE

                    image_labels.append(
                        f'{VIRUS_METADATA.get(virus).get("id")} {yolo_x} {yolo_y} {yolo_w} {yolo_h} \n'
                    )

                else:
                    x_coordinates = [p[0] for p in f_particle]
                    y_coordinates = [p[1] for p in f_particle]

                    margin = VIRUS_METADATA.get(virus)['diameter']
                    min_x = min(x_coordinates) - margin
                    max_x = max(x_coordinates) + margin
                    min_y = min(y_coordinates) - margin
                    max_y = max(y_coordinates) + margin
                    xy = (min_x, min_y)
                    w = max_x - min_x
                    h = max_y - min_y

                    yolo_x = xy[0] + w / 2
                    yolo_y = xy[1] + h / 2
                    yolo_w = w
                    yolo_h = h

                    if not check_box_in_pic(yolo_x, yolo_y, yolo_w, yolo_h):
                        continue

                    #Normalize x,y, w and h
                    yolo_x = yolo_x / YOLO_IMAGE_SIZE
                    yolo_y = yolo_y / YOLO_IMAGE_SIZE
                    yolo_h = yolo_h / YOLO_IMAGE_SIZE
                    yolo_w = yolo_w / YOLO_IMAGE_SIZE

                    image_labels.append(
                        f'{VIRUS_METADATA.get(virus).get("id")} {yolo_x} {yolo_y} {yolo_w} {yolo_h} \n'
                    )

            #if at least one particle makes it, save the image and the file
            if (len(image_labels) > 0):
                crop_img = (
                    cv2.normalize(crop_img, None, 1.0, 0.0, cv2.NORM_MINMAX) *
                    255).astype(np.uint8)

                #save the cropped image
                if split_set == "validation":
                    split_set_save = "valid"
                else:
                    split_set_save = split_set

                saved_image_path = os.path.join(
                    YOLO_DATA_PATH, 'images', split_set_save,
                    f'{virus}_{image_file_name}.jpg')
                cv2.imwrite(saved_image_path, crop_img)

                #save the label file
                with open(
                        os.path.join(YOLO_DATA_PATH, 'labels', split_set_save,
                                     f'{virus}_{image_file_name}.txt'),
                        "w") as f:
                    f.writelines(image_labels)
