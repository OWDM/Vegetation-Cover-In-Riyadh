from osgeo import gdal
import numpy as np
import os
from PIL import Image

# Define the input and output directories
input_dir = "N:\\My Drive\\RUH_all_bands"
output_dir = "N:\\My Drive\\Riyadh_Dataset"


# Create the output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# List all tif files in the input directory
image_files = [f for f in os.listdir(input_dir) if f.endswith('.tif')]

# Function to normalize arrays
def normalize(array):
    array_min, array_max = array.min(), array.max()
    return (array - array_min) / (array_max - array_min) if array_max - array_min > 0 else array

# Function to divide the image into smaller parts
def divide_image(image, output_dir, base_filename):
    height, width, _ = image.shape
    part_width = width // 24
    part_height = height // 24
    for i in range(24):
        for j in range(24):
            part = image[i * part_height:(i + 1) * part_height, j * part_width:(j + 1) * part_width]
            part_image = Image.fromarray(part)
            part_filename = f"{base_filename}_sub_image_{i*24+j+1}.png"
            part_image.save(os.path.join(output_dir, part_filename))

# Process each image file
for img_file in image_files:
    dataset = gdal.Open(os.path.join(input_dir, img_file))

    # Read RGB bands (assuming bands 4, 3, 2 are RGB)
    red = dataset.GetRasterBand(4).ReadAsArray().astype(float)
    green = dataset.GetRasterBand(3).ReadAsArray().astype(float)
    blue = dataset.GetRasterBand(2).ReadAsArray().astype(float)

    # Handle NaN and infinite values
    red[np.isnan(red) | np.isinf(red)] = 0
    green[np.isnan(green) | np.isinf(green)] = 0
    blue[np.isnan(blue) | np.isinf(blue)] = 0

    # Normalize and convert to 8-bit
    red_8bit = (normalize(red) * 255).astype(np.uint8)
    green_8bit = (normalize(green) * 255).astype(np.uint8)
    blue_8bit = (normalize(blue) * 255).astype(np.uint8)

    # Stack to create RGB image
    rgb_8bit = np.stack((red_8bit, green_8bit, blue_8bit), axis=2)

    # Slice the array to fit the division evenly, assuming 6144x6144 is a valid size for all images
    sliced_rgb = rgb_8bit[:6144, :6144]

    # Divide and save images
    base_filename = os.path.splitext(img_file)[0]
    divide_image(sliced_rgb, output_dir, base_filename)

print("All images processed and sub-images saved.")
