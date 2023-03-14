from ddd.params import *


def average_coord(tuple1:tuple, tuple2:tuple):
    new_x = (float(tuple1[0])+ float(tuple2[0]))/2
    new_y = (float(tuple1[1]) + float(tuple2[1]))/2
    new_coord = (new_x,new_y)
    return new_coord



def dictionary_initialization(yolo_image_train_path):
    '''compute images to create for yolo training'''

    dic = {}
    nb_to_create = {}

    for virus in VIRUSES:
        dic[virus] = 0

    for file in os.listdir(yolo_image_train_path):
        ind = file.find('_')
        virus_name = file[:ind]
        dic[virus_name] = dic[virus_name] +1

    for virus in VIRUSES:
        nb_to_create[virus] = 100 - dic[virus]

    #removing unuseful virus
    del nb_to_create['Influenza']
    del nb_to_create['Lassa']
    del nb_to_create['Nipah virus']

    return nb_to_create
