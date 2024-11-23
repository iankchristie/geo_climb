import rasterio

# Open the TIF file using Rasterio
with rasterio.open('D:\GeoClimb\geo_climb\data\labeled\sentinel2\sen_31.9222_-106.0434.tif') as src:
    image = src.read()  # This returns a 3D array (bands, height, width)

# Flatten the image
flattened_image = image.flatten()

print(flattened_image.shape)  # Output shape: (bands * height * width,)
print(flattened_image)
print(type(flattened_image))