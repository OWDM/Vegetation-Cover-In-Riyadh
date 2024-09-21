from osgeo import gdal
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

dataset = gdal.Open(r"H:\My Drive\Riyadh_folder\Riyadh_Median_2021-06-15_to_2021-06-29.tif")

# Read the bands - assuming the bands are 4, 3, 2 for RGB
red = dataset.GetRasterBand(3).ReadAsArray().astype(float)
green = dataset.GetRasterBand(2).ReadAsArray().astype(float)
blue = dataset.GetRasterBand(1).ReadAsArray().astype(float)

# Replace any potential NaN ort infinite values with zeros
red[np.isnan(red) | np.isinf(red)] = 0
green[np.isnan(green) | np.isinf(green)] = 0
blue[np.isnan(blue) | np.isinf(blue)] = 0

# Normalize the bands
def normalize(array):
    array_min, array_max = array.min(), array.max()
    # Avoid division by zero
    return (array - array_min) / (array_max - array_min) if array_max - array_min > 0 else array

red_normalized = normalize(red)
green_normalized = normalize(green)
blue_normalized = normalize(blue)

# Convert the normalized arrays to 8-bit (0-255)
red_8bit = (red_normalized * 255).astype(np.uint8)
green_8bit = (green_normalized * 255).astype(np.uint8)
blue_8bit = (blue_normalized * 255).astype(np.uint8)

# Stack the bands along the third dimension to create an RGB image
rgb_8bit = np.stack((red_8bit, green_8bit, blue_8bit), axis=2)

# Create a PIL image from the numpy array
image = Image.fromarray(rgb_8bit)

# Save the image to a file
#image.save("H:\\My Drive\\rgb new")  # for PNG format

# Display the image using PIL's show() method
image.show()
