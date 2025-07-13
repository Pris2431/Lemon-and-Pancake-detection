# building the cnn model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set paths (update as needed)
train_dir = 'C:\Python\group 5 mlai pancakes/train/'
valid_dir = 'C:\Python\group 5 mlai pancakes/valid/'
test_dir  = 'C:\Python\group 5 mlai pancakes/test/'

# Image sizes (keep consistent!)
IMG_SIZE = (75, 75)
BATCH_SIZE = 32

# Data generators (add augmentations if you like)
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.15,
                                   width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15, horizontal_flip=True)
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen  = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
)
valid_gen = valid_datagen.flow_from_directory(
    valid_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
)
test_gen = test_datagen.flow_from_directory(
    test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
)

# Define your custom CNN model (simple, you can expand if you wish)
model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(3, activation='softmax')  # 3 classes: pancake, lemon, nothing
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Print model summary
model.summary()

# --- ADD DIAGNOSTICS HERE ---
print("---- DATASET CHECK ----")
print("Training images found:", train_gen.samples)
print("Validation images found:", valid_gen.samples)
print("Test images found:", test_gen.samples)
print("Class indices:", train_gen.class_indices)
print("-----------------------")

# Train the model
EPOCHS = 20
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=valid_gen
)

model.save('v1_model.hdf5')