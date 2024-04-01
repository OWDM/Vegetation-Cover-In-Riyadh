from osgeo import gdal
image_path = "C:\\Users\\Musae\\Documents\\GitHub-REPOs\\Senior_Project\\Doc\\random.tif"

ds = gdal.Open(image_path)

# Get raster size
x_size = ds.RasterXSize
y_size = ds.RasterYSize
print(f"Size: {x_size} x {y_size}")
