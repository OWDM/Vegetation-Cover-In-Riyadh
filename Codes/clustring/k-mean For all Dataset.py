import numpy as np
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from osgeo import gdal
from sklearn.cluster import MiniBatchKMeans

def load_and_preprocess_image(image_path):
    sub_image = gdal.Open(image_path)
    NDVI = sub_image.GetRasterBand(1)
    NDVI_array = NDVI.ReadAsArray()
    return np.nan_to_num(NDVI_array, nan=0.01).flatten()

def load_images_from_folder(folder):
    all_pixels = []
    image_shapes = []
    for filename in os.listdir(folder):
        if filename.endswith(".tif"):
            img_path = os.path.join(folder, filename)
            pixels = load_and_preprocess_image(img_path)
            all_pixels.append(pixels)
            image_shapes.append(gdal.Open(img_path).ReadAsArray().shape)
    return np.concatenate(all_pixels), image_shapes


def cluster_global_data(pixels, optimal_k):
    # Use MiniBatchKMeans for clustering
    kmeans = MiniBatchKMeans(n_clusters=optimal_k, random_state=0, batch_size=10000)  # Adjust the batch_size as needed
    kmeans.fit(pixels.reshape(-1, 1))
    return kmeans, kmeans.labels_


def assign_clusters_to_images(global_labels, image_shapes):
    start = 0
    clustered_images = []
    for shape in image_shapes:
        size = shape[0] * shape[1]
        labels = global_labels[start:start + size].reshape(shape)
        clustered_images.append(labels)
        start += size
    return clustered_images

def calculate_cluster_percentages(labels):
    unique, counts = np.unique(labels, return_counts=True)
    total_pixels = labels.size
    percentages = {k: (count / total_pixels) * 100 for k, count in zip(unique, counts)}
    return percentages

def print_cluster_value_ranges(kmeans, pixels):
    for i, center in enumerate(kmeans.cluster_centers_):
        cluster_pixels = pixels[kmeans.labels_ == i]
        print(f'Cluster {i}: range ({cluster_pixels.min():.3f} - {cluster_pixels.max():.3f})')

# Path to your folder containing the images
folder_path = 'C:\\Users\\Musae\\Documents\\GitHub-REPOs\\Senior-project-main\\Docs\\sub from all'
# Load and concatenate pixels from all images
all_pixels, image_shapes = load_images_from_folder(folder_path)

# Optimal number of clusters
optimal_k = 5

# Perform global clustering
kmeans, global_labels = cluster_global_data(all_pixels, optimal_k)

# Calculate and print cluster percentages
percentages = calculate_cluster_percentages(global_labels)
for cluster, percentage in percentages.items():
    print(f"Cluster {cluster}: {percentage:.2f}%")

# Print cluster value ranges
print_cluster_value_ranges(kmeans, all_pixels.reshape(-1, 1))

# Assign clusters back to original images
clustered_images = assign_clusters_to_images(global_labels, image_shapes)# Load and concatenate pixels from all images
all_pixels, image_shapes = load_images_from_folder(folder_path)

# Optimal number of clusters
optimal_k = 5

# Perform global clustering
kmeans, global_labels = cluster_global_data(all_pixels, optimal_k)

# Calculate and print cluster percentages
percentages = calculate_cluster_percentages(global_labels)
for cluster, percentage in percentages.items():
    print(f"Cluster {cluster}: {percentage:.2f}%")

# Print cluster value ranges
print_cluster_value_ranges(kmeans, all_pixels.reshape(-1, 1))

# Assign clusters back to original images
clustered_images = assign_clusters_to_images(global_labels, image_shapes)