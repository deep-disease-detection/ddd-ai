import os
from ddd.params import *
import csv
from ddd.utils import average_coord
import numpy as np
import cv2
import tensorflow as tf


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


def save_imagettes(imagettes: list, split_set: str, virus: str, file: str):
    for i, img in enumerate(imagettes):
        imagette_file_name = f'{file}_{str(i)}.tif'
        destination_path = os.path.join(PROCESS_DATA_PATH, split_set, virus,
                                        imagette_file_name)
        cv2.imwrite(destination_path, img)


def preprocess_viruses(split_set: str = 'train', mode: str = "testing"):

    #Loading structure and data dictionnaries
    particle_dict = get_dic_center()
    pic_dict = get_pic_mesure()

    #Subsetting to specific folders if test mode
    if mode == "testing":
        test_subset = ['Adenovirus']
        print(f'Executing in test mode for {test_subset}')
        particle_dict = {k: particle_dict.get(k) for k in test_subset}
        pic_dict = {k: pic_dict.get(k) for k in test_subset}

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
            img = np.expand_dims(
                cv2.imread(image_path, -1).astype(np.float32), 2)

            #Resize the image
            img = resize_image(img, file, pic_dict[virus])

            #Apply the same transformation to the particles
            resized_particles = resize_particles(particles, file,
                                                 pic_dict[virus])

            #Min-max scale the image
            img = cv2.normalize(img, None, 1.0, 0.0, cv2.NORM_MINMAX)

            #Add padding to the image
            img = cv2.copyMakeBorder(img, IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE,
                                     IMAGE_SIZE, cv2.BORDER_REFLECT)
            img = np.expand_dims(img,
                                 2)  #reshape to (H, W, 1) for tensforflow :)

            #Crop around each particle
            imagettes = make_imagettes(img, resized_particles)

            #Create the files
            save_imagettes(imagettes, split_set, virus, file)


if __name__ == "__main__":
    preprocess_viruses("train", "testing")
