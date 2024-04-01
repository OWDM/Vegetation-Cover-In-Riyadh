import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from osgeo import gdal
sub_image = gdal.Open('C:\\Users\\Musae\\Documents\\GitHub-REPOs\\Senior-project-main\\Docs\\NDVI-Data\\NDVI_RUH-B4-B8_2023-12-19.tif')

NDVI = sub_image.GetRasterBand(1)
NDVI_array = NDVI.ReadAsArray()
ndvi_processed_all = np.nan_to_num(NDVI_array, nan=0.01)

def cluster_image(ndvi_processed_all, optimal_k):
    """
    Clusters the image using K-Means and reshapes the labels to the original image shape.
    Handles both 2D and 3D images.
    """
    if ndvi_processed_all.ndim == 3:
        original_shape = ndvi_processed_all.shape[:2]
        pixels = ndvi_processed_all.reshape(-1, ndvi_processed_all.shape[2])
    elif ndvi_processed_all.ndim == 2:
        original_shape = ndvi_processed_all.shape
        pixels = ndvi_processed_all.flatten().reshape(-1, 1)
    else:
        raise ValueError("Unsupported image dimensionality. Image must be 2D or 3D.")

    kmeans = KMeans(n_clusters=optimal_k, random_state=0).fit(pixels)
    return kmeans, kmeans.labels_.reshape(original_shape)

def plot_clustered_image(clustered_img, optimal_k):
    """
    Plots the clustered image.
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(clustered_img, cmap='RdYlGn')
    plt.title(f'Image clustered into {optimal_k} colors')
    plt.axis('off')
    plt.show()

def plot_cluster_intensity_distributions(kmeans_labels, img_array, optimal_k):
    """
    Plots the distribution of pixel intensities in each cluster.
    """
    plt.figure(figsize=(15, 5))
    for i in range(optimal_k):
        cluster_pixels = img_array[kmeans_labels == i]
        plt.subplot(1, optimal_k, i+1)
        plt.hist(cluster_pixels, bins=30, color='gray')
        plt.title(f'Cluster {i} Intensity Distribution')
        plt.xlabel('Intensity')
        plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

def visualize_clusters_spatially(kmeans_labels, optimal_k):
    """
    Visualizes the spatial distribution of clusters.
    """
    fig, axes = plt.subplots(1, optimal_k, figsize=(15, 5))
    for i in range(optimal_k):
        cluster_img = np.zeros(kmeans_labels.shape)
        cluster_img[kmeans_labels == i] = 1
        axes[i].imshow(cluster_img, cmap='gray')
        axes[i].set_title(f'Cluster {i}')
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

def calculate_cluster_percentages(kmeans_labels):
    """
    Calculates and prints the percentage of pixels in each cluster.
    """
    total_pixels = kmeans_labels.size
    unique, counts = np.unique(kmeans_labels, return_counts=True)
    percentages = {k: count / total_pixels * 100 for k, count in zip(unique, counts)}
    return percentages

def print_cluster_value_ranges(kmeans_labels, img_array, optimal_k):
    """
    Prints the value ranges for each cluster.
    """
    for i in range(optimal_k):
        cluster_pixels = img_array[kmeans_labels == i]
        min_value, max_value = cluster_pixels.min(), cluster_pixels.max()
        print(f'Cluster {i}: range ({min_value:.3f} - {max_value:.3f})')

# Example usage
optimal_k = 4
kmeans, kmeans_labels = cluster_image(ndvi_processed_all, optimal_k)
plot_clustered_image(kmeans_labels, optimal_k)
plot_cluster_intensity_distributions(kmeans_labels, ndvi_processed_all, optimal_k)
visualize_clusters_spatially(kmeans_labels, optimal_k)
percentages = calculate_cluster_percentages(kmeans_labels)
for cluster, percentage in percentages.items():
    print(f"Cluster {cluster}: {percentage:.2f}%")
print_cluster_value_ranges(kmeans_labels, ndvi_processed_all, optimal_k)
