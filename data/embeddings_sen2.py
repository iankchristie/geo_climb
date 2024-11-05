import os
import sys
import torch
import torchgeo
from torchgeo.models import ResNet50_Weights
from torchvision import transforms
import numpy as np
import rasterio
import timm
from torchgeo.models import ResNet50_Weights
from PIL import Image
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config
from downloaders.file_utils import *

weights = ResNet50_Weights.SENTINEL2_RGB_MOCO

model = timm.create_model("resnet50", in_chans=weights.meta["in_chans"], num_classes=10)
model.load_state_dict(weights.get_state_dict(progress=True), strict=False)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove the last layer
model.eval()

sentinel2_transforms = transforms.Compose([
    transforms.Resize(256),  # Resize the shorter side to 256 pixels
    transforms.CenterCrop((224, 224)),  # Crop to 224x224 pixels
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=0, std=10000),  # Normalize with mean=0 and std=10000
])

# Function to load and preprocess Sentinel-2 .tif images
def process_sentinel2_image(image_path):
    with rasterio.open(image_path) as src:
        image = src.read([1, 2, 3])  # Reading RGB channels (bands 1, 2, 3)
        image = np.moveaxis(image, 0, -1)  # Change from (channels, height, width) to (height, width, channels)
    
    # Convert to PIL Image for compatibility with torchvision transforms
    image_pil = Image.fromarray(image.astype(np.uint8))
    transformed_image = sentinel2_transforms(image_pil)
    
    return transformed_image.unsqueeze(0)  # Add batch dimension

def generate_N_save_embeddings(data:str,embeddings_dir:str):
    for sen_file in os.listdir(data):
        os.makedirs(embeddings_dir,exist_ok=True)
        if sen_file.endswith('.tif'):
            lat,lon=decode_file(sen_file)
            image_tensor = process_sentinel2_image(os.path.join(data,sen_file))  
            with torch.no_grad():
                embeddings = model(image_tensor).squeeze()
            
            torch.save(embeddings, encode_file(lat,lon,'unlabelembds',embeddings_dir,'pt'))

if __name__=="__main__":
    generate_N_save_embeddings(Config.DATA_DIR_LBL_SEN,Config.DATA_DIR_LBL_SEN_EMBEDS)
    generate_N_save_embeddings(Config.DATA_DIR_UNLBL_SEN,Config.DATA_DIR_UNLBL_SEN_EMBEDS)