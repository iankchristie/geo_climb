import rasterio
import os
import sys
from preprocess import *
from typing import Literal

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config
from utils.file_utils import *

def flatten_N_save(type: Literal["sen", "dem"], data: str, dir: str):
    """
    Flatten and save the given data.

    Args:
        type (str): Type of the data, either "SEN" or "DEM".
        data (str): Path or reference to the data to be processed.
        dir (str): Directory where the flattened data should be saved.

    """
    for file in os.listdir(data):
        output_dir=os.path.join(dir,f"{data}_flattened")
        os.makedirs(output_dir, exist_ok=True)
        if file.endswith(".tif"):
            lat, lon = decode_file(file)
            with rasterio.open(os.path.join(data, file)) as src:
                image = src.read()  # This returns a 3D array (bands, height, width)
            if  type=="DEM":    
                image_tensor = preprocess_dem(image)
            else:
                image_tensor=preprocess_sen(image)
            flattened_image = image_tensor.numpy().flatten()
            print(flattened_image.shape)
            np.save(
                encode_file(lat, lon, type, output_dir, "npy"), flattened_image
            )


if __name__=="__main__":
    flatten_N_save("dem",Config.DATA_DIR_LBL_DEM,Config.DATA_DIR_LBL_EMBEDS)
    flatten_N_save("sen",Config.DATA_DIR_LBL_SEN,Config.DATA_DIR_LBL_EMBEDS)
    flatten_N_save("dem",Config.DATA_DIR_UNLBL_DEM,Config.DATA_DIR_UNLBL_EMBEDS)
    flatten_N_save("sen",Config.DATA_DIR_UNLBL_SEN,Config.DATA_DIR_UNLBL_EMBEDS)