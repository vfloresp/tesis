import os

DATASET_PATH = ""
JSON_PATH = ""

def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    #dictionary to store data
    data = {
        "mapping":[],
        "mfcc":[],
        "labels":[]
    }

    #loop trhough all genres
    for i,(dirpath, drinames, filenames) in enumerate(os.walk(dataset_path)):
        #ensure that were not at root level
        if dirpath is not dataset_path:
            pass