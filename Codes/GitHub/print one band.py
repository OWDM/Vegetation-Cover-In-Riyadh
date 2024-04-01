from osgeo import gdal
import matplotlib.pyplot as plt


image_path = "C:\\Users\\Musae\\Documents\\GitHub-REPOs\\Senior_Project\\Doc\\random.tif"

ds = gdal.Open(image_path)


band = ds.GetRasterBand(1)
array = band.ReadAsArray()
plt.figure()
plt.imshow(array)
plt.show()
