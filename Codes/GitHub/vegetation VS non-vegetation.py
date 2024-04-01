cluster_counts = [0, 0]

# Loop over each pixel in the NDVI array
for row in range(array.shape[0]):  # Replace 121 with ndvi.shape[0] if it varies
    for col in range(array.shape[1]):  # Replace 82 with ndvi.shape[1] if it varies
        value = array[row, col]
        # Increment the appropriate counter based on the NDVI value
        if value > 0.22:
            cluster_counts[0] += 1
        else:
            cluster_counts[1] += 1

# Plotting the cluster counts
print(cluster_counts)

# Assuming 'cluster_counts' contains the counts for vegetation and land cover
total_count = sum(cluster_counts)
percentages = [(count / total_count) * 100 for count in cluster_counts]

# Cluster labels
cluster_labels = ['Vegetation Cover', 'Land Cover']

# Colors for each cluster
colors = ['green', '#573130']

# Creating the bar chart
plt.figure(figsize=(10, 6))
plt.bar(cluster_labels, percentages, color=colors)

# Adding title and labels
plt.title('Riyadh vegetation cover')
#plt.xlabel('Cluster Type')
plt.ylabel('Percentage of Total Cover (%)')

# Add percentage values above bars
for i, percentage in enumerate(percentages):
    plt.text(i, percentage + 0.5, f'{percentage:.2f}%', ha='center')

# Display the plot
plt.show()