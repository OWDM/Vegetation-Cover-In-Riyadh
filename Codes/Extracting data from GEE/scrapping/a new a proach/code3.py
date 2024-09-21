import ee
from geopy.geocoders import Nominatim
import datetime

# Initialize Earth Engine library
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

            # Original location coordinates
            longitude = location.longitude
            latitude = location.latitude

            # Shift the latitude upward by delta_latitude degrees
            delta_latitude = 0.1  # Adjust this value as needed
            shifted_latitude = latitude + delta_latitude

            # Create an ee.Geometry.Point with adjusted coordinates
            point = ee.Geometry.Point([longitude, shifted_latitude])

            # Define the AOI by buffering the point with the desired distance
            distance = 30720  # Buffer radius in meters (30,720 meters for 61,440 meters side length)
            aoi = point.buffer(distance).bounds()
            print(f"Using square bounding box geometry for {place_name} with side length {distance*2/1000} km")

# Define the bands to select, including 'QA60' for cloud masking
bands = ['B2', 'B3', 'B4', 'B8', 'QA60']

# Helper function to generate two-week intervals
def generate_biweekly_dates(start_date, end_date):
    dates = []
    current_date = start_date
    while current_date < end_date:
        mid_date = current_date + datetime.timedelta(days=13)  # 14 days total including current day
        if mid_date > end_date:
            mid_date = end_date
        dates.append((current_date, mid_date))
        current_date = mid_date + datetime.timedelta(days=1)  # Add 1 day to avoid overlap
    return dates

# Define the start and end date for filtering
start_date = datetime.date(2019, 1, 1)
end_date = datetime.date(2024, 8, 20)

# Generate bi-weekly periods between start_date and end_date
biweekly_periods = generate_biweekly_dates(start_date, end_date)

# Loop through each two-week period and process the images
for period_start, period_end in biweekly_periods:
    # Convert the dates to strings for Earth Engine filtering
    period_start_str = period_start.strftime('%Y-%m-%d')
    period_end_str = period_end.strftime('%Y-%m-%d')
    
    # Filter the Sentinel-2 ImageCollection for the current two-week period and AOI
    sentinelCollection = ee.ImageCollection("COPERNICUS/S2_SR") \
        .filterDate(period_start_str, period_end_str) \
        .filterBounds(aoi) \
        .select(bands)
    
    # Compute the median image for the current two-week period
    median_image = sentinelCollection.median()

    # Clip the median image to the AOI
    clipped_image = median_image.clip(aoi)

    # Cast all bands to Float32
    casted_image = clipped_image.toFloat()

    # Export the image to Drive
    task = ee.batch.Export.image.toDrive(**{
        'image': casted_image,
        'description': f'{place_name}_Median_{period_start_str}_to_{period_end_str}',
        'folder': f'{place_name}_folder',
        'scale': 10,
        'region': aoi.getInfo()['coordinates'],
        'fileFormat': 'GeoTIFF',
        'maxPixels': 1e13
    })
    task.start()
    print(f"Exporting image for period: {period_start_str} to {period_end_str}")

print("All image export tasks started.")
