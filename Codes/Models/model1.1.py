import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from skimage.io import imread
from skimage.transform import resize
import os

# Set paths to the images and masks
image_dir = 'N:\\My Drive\\Data\\RUH'
mask_dir = 'N:\\My Drive\\Data\\Mask'

# List of image and mask files
image_files = sorted(os.listdir(image_dir))[:10000]   # First 15k files
mask_files = sorted(os.listdir(mask_dir))[:10000]

# Function to normalize images and encode masks
def prepare_data(img_path, mask_path):
    # Read the image and mask files
    img = imread(img_path) / 255.0  # Normalize to [0, 1]
    mask = imread(mask_path, as_gray=True)  # Read mask as grayscale

    # Resize images and masks if not already 256x256
    if img.shape[0] != 256 or img.shape[1] != 256:
        img = resize(img, (256, 256), anti_aliasing=True)
    if mask.shape[0] != 256 or mask.shape[1] != 256:
        mask = resize(mask, (256, 256), order=0, preserve_range=True)

    # Map mask pixel values to class labels
    mask[mask == 255] = 3
    mask[mask == 170] = 2
    mask[mask == 85] = 1
    mask[mask == 0] = 0

    # Convert mask to categorical
    mask = to_categorical(mask, num_classes=4)
    
    return img, mask

# Load and prepare the dataset
images = []
masks = []
for img_file, mask_file in zip(image_files, mask_files):
    img_path = os.path.join(image_dir, img_file)
    mask_path = os.path.join(mask_dir, mask_file)
    img, mask = prepare_data(img_path, mask_path)
    images.append(img)
    masks.append(mask)

# Convert to numpy arrays
images = np.array(images)
masks = np.array(masks)


from sklearn.model_selection import train_test_split

# Define the proportion for splitting
train_size = 0.8  # 80% of the data for training
validation_size = 0.1  # 10% of the training data for validation

# First, split into train and test sets
train_images, test_images, train_masks, test_masks = train_test_split(
    images, masks, train_size=train_size, random_state=42
)

# Further split the training set to obtain a validation set
train_images, val_images, train_masks, val_masks = train_test_split(
    train_images, train_masks, test_size=validation_size / train_size, random_state=42
)

# Check the sizes of each dataset part
print(f"Training set: {len(train_images)} images")
print(f"Validation set: {len(val_images)} images")
print(f"Testing set: {len(test_images)} images")


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam

def unet_model(input_size=(256, 256, 3), num_classes=4):  # Adjust input_size to include 3 channels for RGB
    inputs = Input(input_size)

    # Encoder
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    # Bottleneck
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    # Decoder
    u6 = UpSampling2D((2, 2))(c5)
    u6 = Concatenate()([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = UpSampling2D((2, 2))(c6)
    u7 = Concatenate()([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = UpSampling2D((2, 2))(c7)
    u8 = Concatenate()([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = UpSampling2D((2, 2))(c8)
    u9 = Concatenate()([u9, c1])
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    # Output layer
    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Create the U-Net model
unet = unet_model()

# Display the model architecture
unet.summary()


from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Compile the model
unet.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Define callbacks for saving the model and early stopping
model_checkpoint = ModelCheckpoint('unet_segmentation.keras', monitor='val_loss', save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# Training the model
history = unet.fit(
    x=train_images,
    y=train_masks,
    batch_size=32,
    epochs=30,
    validation_data=(val_images, val_masks),
    callbacks=[model_checkpoint, early_stopping]
)

# Optional: Plot training history for loss and accuracy
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
