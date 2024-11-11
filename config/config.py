import os
from dotenv import load_dotenv

load_dotenv()

DATA_DIR_LBL_SEN = os.getenv("DATA_DIR_LBL_SEN")
DATA_DIR_LBL_DEM = os.getenv("DATA_DIR_LBL_DEM")
DATA_DIR_LBL_LITH = os.getenv("DATA_DIR_LBL_LITH")
DATA_DIR_UNLBL_SEN_EMBEDS=os.getenv("DATA_DIR_UNLBL_SEN_EMBEDS")
DATA_DIR_LBL_SEN_EMBEDS=os.getenv("DATA_DIR_LBL_SEN_EMBEDS")
DATA_DIR_UNLBL_SEN = os.getenv("DATA_DIR_UNLBL_SEN")
DATA_DIR_UNLBL_DEM = os.getenv("DATA_DIR_UNLBL_DEM")
DATA_DIR_UNLBL_LITH = os.getenv("DATA_DIR_UNLBL_LITH")

DATA_DIR_LBL_LITH_EMB = os.getenv("DATA_DIR_LBL_LITH_EMB")
DATA_DIR_UNLBL_LITH_EMB = os.getenv("DATA_DIR_UNLBL_LITH_EMB")

DATA_DIR_LBL_LITH_EMB_V2 = os.getenv("DATA_DIR_LBL_LITH_EMB_V2")
DATA_DIR_UNLBL_LITH_EMB_V2 = os.getenv("DATA_DIR_UNLBL_LITH_EMB_V2")

DATA_AGGREGATION = os.getenv("DATA_AGGREGATION")
DATA_TRAINING = os.getenv("DATA_TRAINING")
DATA_AGGREGATION = os.getenv("DATA_AGGREGATION")
DATA_VALIDATION = os.getenv("DATA_VALIDATION")
DATA_TEST = os.getenv("DATA_TEST")

DATA_AGGREGATION_V2 = os.getenv("DATA_AGGREGATION_V2")
DATA_TRAINING_V2 = os.getenv("DATA_TRAINING_V2")
DATA_VALIDATION_V2 = os.getenv("DATA_VALIDATION_V2")
DATA_TEST_V2 = os.getenv("DATA_TEST_V2")


class Config:
    DATA_DIR_LBL_SEN = DATA_DIR_LBL_SEN
    DATA_DIR_LBL_DEM = DATA_DIR_LBL_DEM
    DATA_DIR_LBL_LITH = DATA_DIR_LBL_LITH
    DATA_DIR_UNLBL_SEN_EMBEDS=DATA_DIR_UNLBL_SEN_EMBEDS
    DATA_DIR_LBL_SEN_EMBEDS=DATA_DIR_LBL_SEN_EMBEDS
    DATA_DIR_UNLBL_SEN = DATA_DIR_UNLBL_SEN
    DATA_DIR_UNLBL_DEM = DATA_DIR_UNLBL_DEM
    DATA_DIR_UNLBL_LITH = DATA_DIR_UNLBL_LITH

    DATA_DIR_LBL_LITH_EMB = DATA_DIR_LBL_LITH_EMB
    DATA_DIR_UNLBL_LITH_EMB = DATA_DIR_UNLBL_LITH_EMB

    DATA_DIR_LBL_LITH_EMB_V2 = DATA_DIR_LBL_LITH_EMB_V2
    DATA_DIR_UNLBL_LITH_EMB_V2 = DATA_DIR_UNLBL_LITH_EMB_V2

    DATA_AGGREGATION = DATA_AGGREGATION
    DATA_TRAINING = DATA_TRAINING
    DATA_VALIDATION = DATA_VALIDATION
    DATA_TEST = DATA_TEST

    DATA_AGGREGATION_V2 = DATA_AGGREGATION_V2
    DATA_TRAINING_V2 = DATA_TRAINING_V2
    DATA_VALIDATION_V2 = DATA_VALIDATION_V2
    DATA_TEST_V2 = DATA_TEST_V2
