import ee
import datetime

# Initialize the Earth Engine library
ee.Initialize()

# Define ROI and time range
roi = ee.FeatureCollection('your_roi')
start_date = '2018-01-01'
end_date = '2023-12-31'

# Function to create sub-regions
def create_sub_regions(roi, size):
    bounds = roi.geometry().bounds()
    return ee.FeatureCollection.randomPoints(region=bounds, points=1e6, seed=42).map(
        lambda point: point.buffer(size / 2).bounds()
    ).filterBounds(roi)

# Create sub-regions
sub_regions = create_sub_regions(roi, 1000)  # 1km sub-regions

# Function to assess cloud cover for each sub-region
def assess_cloud_cover(image):
    qa = image.select('QA60')
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    cloud_mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
        qa.bitwiseAnd(cirrus_bit_mask).eq(0)
    )
    
    cloud_percentage = cloud_mask.reduceRegions(
        collection=sub_regions,
        reducer=ee.Reducer.mean(),
        scale=20
    )
    
    return cloud_percentage

# Function to create a clean mask based on cloud assessment
def create_clean_mask(cloud_assessment, threshold):
    return cloud_assessment.map(lambda feature: feature.set(
        'isClean', ee.Number(feature.get('mean')).gt(threshold)
    )).filter(ee.Filter.eq('isClean', True))

# Function to process each image
def process_image(image):
    cloud_assessment = assess_cloud_cover(image)
    clean_regions = create_clean_mask(cloud_assessment, 0.95)  # 90% clean threshold
    clean_mask = ee.Image.constant(1).clip(clean_regions)
    
    # Select only the required bands
    image_selected_bands = image.select(['B2', 'B3', 'B4', 'B8'])
    
    return image_selected_bands.updateMask(clean_mask).set({
        'clean_area_percentage': clean_regions.size().divide(sub_regions.size()),
        'acquisition_date': image.date().format('YYYY-MM-dd')
    })

# Get and process Sentinel-2 images
s2_collection = (ee.ImageCollection('COPERNICUS/S2_SR')
    .filterBounds(roi)
    .filterDate(start_date, end_date)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))  # Initial coarse filter
    .select(['B2', 'B3', 'B4', 'B8', 'QA60'])  # Select required bands plus QA60 for cloud masking
    .map(process_image))

# Function to export each image
def export_image(image):
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=f"S2_Clean_BGRNIR_{image.get('acquisition_date').getInfo()}",
        scale=10,
        region=roi,
        maxPixels=1e13
    )
    task.start()
    return task

# Export all processed images
tasks = s2_collection.map(export_image).getInfo()

# Print information about the processed collection
print(f"Number of processed images: {s2_collection.size().getInfo()}")
print("Image collection:", s2_collection.getInfo())

# Optional: Monitor export tasks
for task in tasks:
    print(f"Task {task['id']} for image {task['description']} started.")