from tensorflow.keras.models import load_model
import numpy as np
import os

# Load the saved autoencoder model
autoencoder = load_model('C:\\Users\\Musae\\Documents\\GitHub-REPOs\\Vegetation-Cover-In-Riyadh\\Codes\\Anomaly detection model\\autoencoder.h5')

# Load your training or validation data
# This should be the same data or similar data that you used to train the autoencoder
# Assuming your data is stored in .npy files and 'data_directory' is the path to these files.
data_directory = 'C:\\Users\\Musae\\Documents\\GitHub-REPOs\\Senior-project_Doc\\Docs\\Array\\NDVI-Reduced'
file_list = os.listdir(data_directory)
data = []
# Load each file and append to data list
for file_name in file_list:
    file_path = os.path.join(data_directory, file_name)
    if file_name.endswith('.npy'):
        array = np.load(file_path)
        data.append(array)
# Convert list to NumPy array and reshape if necessary
data = np.array(data)
data = data.reshape(data.shape[0], -1)  # Flatten the data if it's not already

     
# Predict on the data using the autoencoder
reconstructed_data = autoencoder.predict(data)

# Calculate reconstruction errors
reconstruction_errors = np.mean(np.square(data - reconstructed_data), axis=1)

# Determine the threshold, e.g., 99th percentile of the reconstruction errors
threshold = np.percentile(reconstruction_errors, 99)

# Save the threshold for later use
# You could save it to a file or a database, depending on your needs

print(f"Anomaly detection threshold: {threshold}")
