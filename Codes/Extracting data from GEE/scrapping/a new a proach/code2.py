import ee
from geopy.geocoders import Nominatim

# Initialize the Earth Engine library
ee.Initialize()

# Get user input for the location
place_name = input("Enter a city, state, or country: ")

# Try to get the place from the country dataset
countries = ee.FeatureCollection("FAO/GAUL/2015/level0")
country = countries.filter(ee.Filter.eq('ADM0_NAME', place_name))

# Check if the country collection is not empty
if country.size().getInfo() > 0:
    aoi = country.geometry()
    print(f"Using country geometry for {place_name}")
else:
    # Try to get the place from the state dataset
    states = ee.FeatureCollection("FAO/GAUL/2015/level1")
    state = states.filter(ee.Filter.eq('ADM1_NAME', place_name))
    if state.size().getInfo() > 0:
        aoi = state.geometry()
        print(f"Using state geometry for {place_name}")
    else:
        # Assume it's a city and use geocoding to get the location
        geolocator = Nominatim(user_agent="my_app")
        location = geolocator.geocode(place_name)

        if location is None:
            print("Could not geocode the place name.")
            exit()
        else:
            print(f"Location: {location.address}")
            print(f"Latitude: {location.latitude}, Longitude: {location.longitude}")

            # Create an ee.Geometry.Point
            point = ee.Geometry.Point([location.longitude, location.latitude])

            # Adjust the distance to zoom out
            distance = 5000  # Buffer radius in meters (e.g., 50 km)
            aoi = point.buffer(distance).bounds()  # Use bounds to create a square AOI
            print(f"Using square bounding box geometry for {place_name} with side length {distance*2/1000} km")

# Define the bands to select
spectral_bands = ['B2', 'B3', 'B4', 'B8']  # 10m resolution bands
cloud_bands = ['SCL', 'MSK_CLDPRB']  # 20m resolution bands
all_bands = spectral_bands + cloud_bands

# Adjust the date range to dates with available data
start_date = '2019-01-12'
end_date = '2019-04-25'

# Filter the Sentinel-2 ImageCollection for images within the AOI and time frame
sentinelCollection = ee.ImageCollection("COPERNICUS/S2_SR") \
    .filterDate(start_date, end_date) \
    .filterBounds(aoi) \
    .select(all_bands)

# Function to resample 20m bands to 10m resolution
def resample_bands(image):
    # Resample the 20m bands to 10m
    scl_10m = image.select('SCL').resample('bilinear').reproject(crs='EPSG:4326', scale=10)
    cldprb_10m = image.select('MSK_CLDPRB').resample('bilinear').reproject(crs='EPSG:4326', scale=10)
    # Combine the resampled bands with the spectral bands
    return image.select(spectral_bands).addBands([scl_10m, cldprb_10m])

# Apply the resampling function to each image in the collection
sentinelCollection = sentinelCollection.map(resample_bands)

# Compute the median image from the collection
median_image = sentinelCollection.median()

# Clip the median image to the AOI
clipped_image = median_image.clip(aoi)

# Cast all bands to Float32
casted_image = clipped_image.toFloat()

# Export the image to Google Drive
task = ee.batch.Export.image.toDrive(**{
    'image': casted_image,
    'description': f'{place_name}_6_Bands_3',
    'folder': f'{place_name}_folder',
    'scale': 10,
    'region': aoi,
    'fileFormat': 'GeoTIFF',
    'maxPixels': 1e13  # Adjust if necessary
})
task.start()

print("Image export task started.")
