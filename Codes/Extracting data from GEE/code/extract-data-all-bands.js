// Filter the Sentinel-2 ImageCollection for images within the AOI and time frame
var sentinelCollection = ee.ImageCollection("COPERNICUS/S2_SR")
.filterDate('2024-01-01','2024-01-30')
.median();
print(sentinelCollection);

// Clip the image to the AOI
var clippedImage = sentinelCollection.clip(aoi);

// Function to cast all bands to Float32
function castImage(image) {
  var floatImage = image.toFloat();
  return floatImage;
}

// Cast all bands of the clipped image to Float32
var castedImage = castImage(clippedImage);

// Export the casted image, specifying scale and region.
Export.image.toDrive({
  image: castedImage,
  description: 'RUH-24-01-30-fULL-Bands',
  scale: 10,
  region: aoi.toGeoJSON(),
  fileFormat: 'GeoTIFF',
  maxPixels: 1e9
});

// Add the casted image layer to the map so you can visualize it
Map.centerObject(aoi, 10);  // Adjust the zoom level to your AOI
Map.addLayer(castedImage, {max: 3000}, 'Full Sentinel-2 Image Casted to Float32');