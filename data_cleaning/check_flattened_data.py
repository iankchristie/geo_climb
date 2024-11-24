import numpy as np
import os
import sys
from typing import Literal
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config

def loadFlattenedEmbeds(type: Literal["sen",'dem'],dir:str):
    sz=0

    for file in os.listdir(os.path.join(dir,f"{type}_flattened")):
        if file.endswith(".npy"):
            if sz==0:
                sz=np.load(os.path.join(dir,f"{type}_flattened",file)).shape
            else:
                if sz==np.load(os.path.join(dir,f"{type}_flattened",file)).shape:
                    continue
                else:
                    print("Size doen't match!!")

    print(f"Size of flattened {type} data from Dir:{dir} is {sz}")

if __name__=="__main__":
    #loadFlattenedEmbeds("sen",Config.DATA_DIR_LBL_EMBEDS)
    #loadFlattenedEmbeds("dem",Config.DATA_DIR_LBL_EMBEDS)
    #loadFlattenedEmbeds("sen",Config.DATA_DIR_UNLBL_EMBEDS)
    loadFlattenedEmbeds("dem",Config.DATA_DIR_UNLBL_EMBEDS)