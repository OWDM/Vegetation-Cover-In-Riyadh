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
image_files = sorted(os.listdir(image_dir))[:20000]
mask_files = sorted(os.listdir(mask_dir))[:20000]

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

# Generator to load images in batches
def generate_batches(image_files, mask_files, batch_size):
    num_samples = len(image_files)
    while True:  # Loop indefinitely
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            images, masks = [], []
            for i in range(start, end):
                img_path = os.path.join(image_dir, image_files[i])
                mask_path = os.path.join(mask_dir, mask_files[i])
                img, mask = prepare_data(img_path, mask_path)
                images.append(img)
                masks.append(mask)
            yield np.array(images), np.array(masks)



from sklearn.model_selection import train_test_split
# Split the dataset into training and validation sets
image_files_train, image_files_val, mask_files_train, mask_files_val = train_test_split(
    image_files, mask_files, test_size=0.2, random_state=42)

# Define batch size
batch_size = 32

# Training and validation generator
train_generator = generate_batches(image_files_train, mask_files_train, batch_size)
val_generator = generate_batches(image_files_val, mask_files_val, batch_size)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam

def unet_model(input_size=(256, 256, 3), num_classes=4):
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
model_checkpoint = ModelCheckpoint('unet_segmentation_3.keras', monitor='val_loss', save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# Training the model
history = unet.fit(
    x=train_generator,
    y=None,  # Since the generator yields both images and masks, y is not separately provided
    batch_size=None,  # Batch size is handled by the generator
    epochs=20,
    steps_per_epoch=int(np.ceil(len(image_files_train) / batch_size)),  # Convert to int
    validation_data=val_generator,
    validation_steps=int(np.ceil(len(image_files_val) / batch_size)),  # Convert to int
    callbacks=[model_checkpoint, early_stopping]
)
