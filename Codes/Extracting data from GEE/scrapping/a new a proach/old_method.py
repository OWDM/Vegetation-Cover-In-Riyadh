import datetime
# Function to cast all bands to Float32
def castImage(image):
    return image.toFloat()

# Function to export images
def exportImage(image, date):
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=f'RUH-All-Bands_{date.strftime("%Y-%m-%d")}',
        scale=10,
        region=aoi.getInfo()['coordinates'],
        fileFormat='GeoTIFF',
        maxPixels=1e9
    )
    task.start()

# Start and end dates
start_date = datetime.datetime(2018, 12, 15)
end_date = datetime.datetime(2024, 1, 30)

# Generate images every 30 days
current_date = start_date
while current_date <= end_date:
    # Define the date range for the current image
    date_range_start = current_date.strftime('%Y-%m-%d')
    next_date = current_date + datetime.timedelta(days=30)
    date_range_end = next_date.strftime('%Y-%m-%d')

    # Get the image collection for the current date range
    sentinelCollection = ee.ImageCollection("COPERNICUS/S2_SR") \
        .filterDate(date_range_start, date_range_end) \
        .filterBounds(aoi) \
        .map(maskClouds) \
        .median() 
        
    

    # Clip the image to the AOI
    clippedImage = sentinelCollection.clip(aoi)

    # Cast all bands of the clipped image to Float32
    castedImage = castImage(clippedImage)

    # Export the casted image
    exportImage(castedImage, current_date)

    # Update the current date
    current_date = next_date

print("Images export tasks started.")