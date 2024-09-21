import ee
import geemap
import os
from geopy.geocoders import Nominatim

# Initialize Earth Engine
ee.Initialize()

# User inputs
location_name = 'France'  # Replace with your desired city or country name
start_date = '2020-01-01'
end_date = '2020-01-10'
max_cloud_percentage = 5  # Maximum allowed cloud cover percentage
output_folder = 'output_images'  # Folder to save images

# Create output directory if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get the boundary of the city or country
geolocator = Nominatim(user_agent="geoapiExercises")
location = geolocator.geocode(location_name)

if location:
    print(f"Location found: {location.address}")
    # Create a point geometry
    point = ee.Geometry.Point([location.longitude, location.latitude])
    # Define the ROI as a buffer around the point (e.g., 50 km)
    roi = point.buffer(50000)  # Buffer in meters
else:
    raise ValueError('Location not found. Please check the location name.')

# Subdivide the ROI into 1km x 1km grid cells
grid = geemap.fishnet(
    roi=roi,
    cellsize=1000,  # Cell size in meters
    crs='EPSG:3857',  # Coordinate reference system
    geometry_type='Polygon'  # Output geometry type
)

print(f"Total grid cells created: {grid.size().getInfo()}")

# Prepare the Sentinel-2 image collection
collection = (ee.ImageCollection('COPERNICUS/S2_SR')
              .filterDate(start_date, end_date)
              .filterBounds(roi)
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud_percentage))
              .select(['B8', 'B4', 'B3', 'B2']))

print(f"Total images in collection: {collection.size().getInfo()}")

# Function to process each grid cell
def process_cell(cell_feature):
    cell_geom = cell_feature.geometry()
    # Filter the collection to the grid cell
    cell_collection = collection.filterBounds(cell_geom)
    # Get the least cloudy image
    image = cell_collection.sort('CLOUDY_PIXEL_PERCENTAGE').first()
    if image:
        # Crop the image to the cell geometry
        image = image.clip(cell_geom)
        # Define the output filename
        cell_id = cell_feature.id().getInfo()
        filename = f'sentinel_{cell_id}.tif'
        out_path = os.path.join(output_folder, filename)
        # Export the image
        geemap.ee_export_image(
            image=image,
            filename=out_path,
            scale=10,  # Sentinel-2 spatial resolution is 10 meters
            region=cell_geom,
            file_per_band=False
        )
        print(f"Image saved: {filename}")
    else:
        print('No suitable image found for this cell.')

# Iterate over the grid cells
grid_list = grid.toList(grid.size())

for i in range(grid.size().getInfo()):
    cell_feature = ee.Feature(grid_list.get(i))
    process_cell(cell_feature)
