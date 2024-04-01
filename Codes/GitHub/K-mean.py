import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

def cluster_image(sub_image, optimal_k):
    if sub_image.ndim == 3:
        # For 3D arrays: Reshape assuming the third dimension is color channels
        original_shape = sub_image.shape[:2]  # Get the original height and width
        pixels = sub_image.reshape(-1, sub_image.shape[2])  # Reshape to a 2D array for K-Means
    elif sub_image.ndim == 2:
        # For 2D arrays: Use directly for K-Means
        original_shape = sub_image.shape
        pixels = sub_image.flatten().reshape(-1, 1)  # Reshape to 2D array with one feature
    else:
        raise ValueError("Unsupported image dimensionality. Image must be 2D or 3D.")

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=optimal_k, random_state=0).fit(pixels)
    
    # Reshape the labels back to the original image shape
    clustered_img = kmeans.labels_.reshape(original_shape)
    
    # Plotting
    plt.figure(figsize=(10, 10))
    plt.imshow(clustered_img, cmap='RdYlGn')
    plt.title(f'Image clustered into {optimal_k} colors')
    plt.axis('off')  # Hide axes ticks
    plt.show()

sub_image = "C:\\Users\\Musae\\Documents\\GitHub-REPOs\\Senior-project-main\\Codes\\normal codes\\sub_images\\RUH_2018-12-15_0_0.npy"

# Make sure to replace 'sub_image' with your actual image array
cluster_image(sub_image, optimal_k=4)
