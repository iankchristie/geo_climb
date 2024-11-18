from torchvision.transforms import Resize
import numpy as np
import torch
import torch.nn.functional as F


def preprocess_dem(dem_data):
    """
    TODO(iankc): Use interpolate instead of resize here.
    Normalize the DEM data and resize it to (18, 23).
    """
    # Subtract the minimum value to normalize the DEM data
    min_value = np.min(dem_data)
    normalized_data = dem_data - min_value

    # Convert the normalized data to a float32 tensor
    dem_tensor = (
        torch.from_numpy(normalized_data).float().unsqueeze(0)
    )  # Shape: (1, H, W)

    # Resize the tensor to the target size (18, 23)
    resize_transform = Resize((18, 23))
    resized_tensor = resize_transform(dem_tensor.unsqueeze(0)).squeeze(
        0
    )  # Shape: (1, 18, 23)

    return resized_tensor


def preprocess_sen(sen_data):
    """
    Convert the input data to a PyTorch tensor, permute the dimensions,
    and resize it to (3, 52, 66).
    """
    # Convert the input data to a PyTorch tensor and permute dimensions to (C, H, W)
    sen_tensor = torch.tensor(sen_data, dtype=torch.float32).permute(
        2, 0, 1
    )  # Shape: (3, H, W)

    # Resize the tensor to (3, 52, 66) using bilinear interpolation
    sen_tensor = F.interpolate(
        sen_tensor.unsqueeze(0), size=(52, 66), mode="bilinear", align_corners=False
    )

    # Remove the batch dimension
    sen_tensor = sen_tensor.squeeze(0)  # Shape: (3, 52, 66)

    return sen_tensor
