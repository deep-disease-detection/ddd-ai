import os
from ddd.params import *
from ddd.utils import average_coord
import csv



def get_dic_center(split_set:str = set):
    '''
        Create a dictionary with all coordinates center for each virus and each file
    '''
    #dictionary initiation
    train_data_dic = {}
    list_virus = os.listdir(os.path.join(RAW_DATA_PATH, split_set))
    for virus in list_virus:
        train_data_dic[virus] = {}
    #filling the dictinary virus by virus
    for virus in list_virus:
        path_position_file = os.path.join(RAW_DATA_PATH,split_set,virus, 'particle_positions')
        list_position_file = os.listdir(path_position_file)
        #looping over file
        for file in list_position_file:
            with open(os.path.join(RAW_DATA_PATH,split_set,virus, 'particle_positions', file), 'r') as f:
                lines = f.readlines()
                center_coords = []
                particle =[]
                for i in range(3,len(lines)):
                    if lines[i] != 'particleposition\n':
                        coordinate = tuple(lines[i].strip('\n').split(';'))
                        particle.append(coordinate)
                        if i == len(lines)-1:
                            center_coords.append(average_coord(particle[0],particle[1]))
                    else:
                        #compute the center between 2 center point of one single particule
                        if len(particle) == 2:
                            center_coords.append(average_coord(particle[0],particle[1]))
                            particle = []
                        else:
                            for coord in particle:
                                center_coords.append(coord)
                            particle= []
                file_name = file.rstrip('_particlepositions.txt')
                train_data_dic[virus][file_name] = center_coords

    return train_data_dic




def get_pic_mesure(split_set:str = 'train'):
    '''
        return a dictionary with image mesurement and scale
    '''

    pic_data_dic = {}
    list_virus = os.listdir(os.path.join(RAW_DATA_PATH, split_set))
    for virus in list_virus:
        pic_data_dic[virus] =  {}

    for virus in list_virus:
        path_tag_files = os.path.join(RAW_DATA_PATH,split_set,virus, 'tags')
        list_tag_files = os.listdir(path_tag_files)
        for file in list_tag_files:
            with open(f'{path_tag_files}/{file}', 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter=';')
                all_infos = []
                for row in reader:
                    all_infos.append(row)
                file_name = file.rstrip('.tif_tags.csv')
                pic_data_dic[virus][file_name] = {}
                pic_data_dic[virus][file_name]['Height'] = int(all_infos[0][1])
                pic_data_dic[virus][file_name]['Width'] = int(all_infos[1][1])
                pic_data_dic[virus][file_name]['Xscale'] = float(all_infos[5][1].replace(',','.'))
                pic_data_dic[virus][file_name]['Yscale'] = float(all_infos[6][1].replace(',','.'))
    return pic_data_dic
