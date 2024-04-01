def read_band(band_number):
    band = ds.GetRasterBand(band_number).ReadAsArray().astype(np.float32)
    return band

red = read_band(1)  #  Band 1 is 4 (red band) 
nir = read_band(2)  # band 2 is 8 (NIR band)


ndvi = (nir - red) / (nir + red)


# Plot NDVI
plt.figure(figsize=(10, 10))
plt.imshow(ndvi, cmap='RdYlGn')
plt.colorbar(label='NDVI')
plt.axis('off')  # Turn off the axis
plt.show()