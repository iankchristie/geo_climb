import numpy as np
import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config

def convertToTensor(numpy_array):
    return torch.from_numpy(numpy_array)

def convert_N_save(read_dir:str,output_dir:str):
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(read_dir):
        if file.endswith(".npy"):
            newfile=f"{file[:-4]}.pt"
            tensor=convertToTensor(np.load(os.path.join(read_dir,file)))
            print(tensor.shape)
            torch.save(tensor, os.path.join(output_dir,newfile))


if __name__=="__main__":
    #convert_N_save(os.path.join(Config.DATA_DIR_LBL_EMBEDS,"dem_flattened"),os.path.join(Config.DATA_DIR_LBL_EMBEDS,"dem_flattenedTensor"))
    #convert_N_save(os.path.join(Config.DATA_DIR_LBL_EMBEDS,"sen_flattened"),os.path.join(Config.DATA_DIR_LBL_EMBEDS,"sen_flattenedTensor"))
    #convert_N_save(os.path.join(Config.DATA_DIR_UNLBL_EMBEDS,"dem_flattened"),os.path.join(Config.DATA_DIR_UNLBL_EMBEDS,"dem_flattenedTensor"))
    convert_N_save(os.path.join(Config.DATA_DIR_UNLBL_EMBEDS,"sen_flattened"),os.path.join(Config.DATA_DIR_UNLBL_EMBEDS,"sen_flattenedTensor"))