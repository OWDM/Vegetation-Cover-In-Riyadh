import numpy as np
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

def load_data(directory):
    files = os.listdir(directory)
    all_data = []
    for file in files:
        if file.endswith(".npy"):
            data_path = os.path.join(directory, file)
            data = np.load(data_path)
            all_data.append(data)
    return np.array(all_data)

def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    # Encoder
    encoded = Dense(128, activation='relu')(input_layer)
    encoded = Dense(64, activation='relu')(encoded)
    # Decoder
    decoded = Dense(128, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)
    
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return autoencoder

def train_autoencoder(autoencoder, data):
    autoencoder.fit(data, data, epochs=10, batch_size=256, shuffle=True, validation_split=0.1)
    # Save the trained model
    autoencoder.save('C:\\Users\\Musae\\Documents\\GitHub-REPOs\\Vegetation-Cover-In-Riyadh\\Codes\\Anomaly detection model\\autoencoder.h5')

def detect_anomalies(autoencoder, data):
    predictions = autoencoder.predict(data)
    errors = np.mean(np.square(data - predictions), axis=1)
    threshold = np.quantile(errors, 0.99)
    print(f"Anomaly threshold: {threshold}") 
    anomalies = data[errors > threshold]
    return anomalies

# Path to your NDVI data directory
data_directory = r'C:\Users\Musae\Documents\GitHub-REPOs\Senior-project_Doc\Docs\Array\NDVI-Reduced'
data = load_data(data_directory)
data = data.reshape(data.shape[0], -1)  # Flatten data if it's not already

autoencoder = build_autoencoder(data.shape[1])
train_autoencoder(autoencoder, data)
anomalies = detect_anomalies(autoencoder, data)

print(f"Detected {len(anomalies)} anomalies.")
