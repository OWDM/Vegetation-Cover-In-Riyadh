from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

# Starting list of colormaps
colormaps = ['RdYlGn', 'viridis', 'jet', 'Greys', 'terrain',
             'coolwarm', 'BrBG', 'cividis', 'plasma', 'magma',
             'PiYG', 'YlGn', 'BuGn', 'YlGnBu', 'GnBu']

# Number of colormaps dictates the number of subplots
n_colormaps = len(colormaps)
n_cols = 1  # Number of columns per row, set to 1 for one image per row
n_rows = n_colormaps  # One row for each colormap

# Create a figure with the calculated number of subplots (rows and columns)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 6 * n_rows))

# If there's only one column, axes might not be an array if n_rows is 1
if n_rows == 1:
    axes = [axes]  # Make it a list to keep the iteration below consistent

# Plot NDVI with each colormap
for ax, cmap in zip(axes, colormaps):
    im = ax.imshow(ndvi, cmap=cmap)
    ax.set_title(cmap)
    ax.axis('off')  # Remove axis ticks and labels

    # Add a colorbar to each subplot
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

plt.tight_layout()
plt.show()
