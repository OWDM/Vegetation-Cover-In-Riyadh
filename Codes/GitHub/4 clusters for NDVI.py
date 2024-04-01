cluster_counts = [0, 0, 0, 0]

# Loop over each pixel in the NDVI array
for row in range(ndvi.shape[0]):  # Replace 121 with ndvi.shape[0] if it varies
    for col in range(ndvi.shape[1]):  # Replace 82 with ndvi.shape[1] if it varies
        value = ndvi[row, col]
        # Increment the appropriate counter based on the NDVI value
        if value > 0.66:
            cluster_counts[0] += 1
        elif value > 0.33:
            cluster_counts[1] += 1
        elif value > 0.01:
            cluster_counts[2] += 1
        else:
            cluster_counts[3] += 1

# Plotting
clusters = ['> 0.66', '0.33 - 0.66', '0.01 - 0.33', '<= 0.01']
plt.bar(clusters, cluster_counts, color=['green', 'yellow', 'orange', 'red'])
plt.title('NDVI Pixel Distribution Across Clusters')
plt.xlabel('Cluster Range')
plt.ylabel('Pixel Count')
plt.show()
print(cluster_counts)
