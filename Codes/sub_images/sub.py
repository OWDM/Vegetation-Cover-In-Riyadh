from osgeo import gdal
import numpy as np
import os

# Define the path where you want to save the sub-images
output_dir = "G:\\My Drive\\RUH_One_sub-4"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Open the image
img_path = gdal.Open("G:\\My Drive\\RUH_all_bands\\RUH-All-Bands_2018-12-15.tif")
NDVI = img_path.GetRasterBand(4)
ndvi = NDVI.ReadAsArray()

# Function to divide the image into smaller parts
def divide_image(image):
    height, width = image.shape
    part_width = width // 24
    part_height = height // 24
    parts = []
    for i in range(24):
        for j in range(24):
            part = image[i * part_height : (i + 1) * part_height, j * part_width : (j + 1) * part_width]
            parts.append(part)
    return parts

# Slice the array to fit the division evenly
sliced_array = ndvi[:6144, :6144]
img_parts = divide_image(sliced_array)

# Save each part as a NumPy array
for index, img_part in enumerate(img_parts):
    # Create a filename for each sub-image
    filename = f"sub_image_{index + 1}.npy"
    filepath = os.path.join(output_dir, filename)
    np.save(filepath, img_part)

print("All sub-images have been saved successfully as NumPy arrays.")
