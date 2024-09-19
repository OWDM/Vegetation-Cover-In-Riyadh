import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal

# Open the red band data
data_red = "G:\\My Drive\\RUH_One_sub-4\\sub_image_294.tif"
red_ds = gdal.Open(data_red)

# Open the NIR band data
data_nir = "G:\\My Drive\\RUH_One_sub-8\\sub_image_294.tif"
nir_ds = gdal.Open(data_nir)

# Read bands and convert to float32 for calculation
red = red_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
nir = nir_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)

# Calculate NDVI
ndvi = (nir - red) / (nir + red)

# Plot NDVI
plt.figure(figsize=(10, 10))
plt.imshow(ndvi, cmap='RdYlGn')
plt.colorbar(label='NDVI')
plt.axis('off')  # Turn off the axis
plt.show()