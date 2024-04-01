import os
from osgeo import gdal
import matplotlib.pyplot as plt

# Define the folder path
folder_path = "G:\\My Drive\\Riyadh-Data-set"

# Iterate over each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.tif'):  # Assuming the files are GeoTIFFs; adjust the extension as necessary
        file_path = os.path.join(folder_path, filename)
        
        # Open the file with GDAL
        dataset = gdal.Open(file_path)
        
        if dataset is None:
            print(f"Failed to open file {filename}")
            continue
        
        # Read the first band
        band = dataset.GetRasterBand(1)  # Assuming you want the first band
        arr = band.ReadAsArray()
        
        # Plotting the band
        plt.figure(figsize=(10, 10))
        plt.imshow(arr, cmap='gray')
        plt.title(f'{filename} - Band 1')
        plt.colorbar()
        plt.show()

        # Close the dataset
        dataset = None

