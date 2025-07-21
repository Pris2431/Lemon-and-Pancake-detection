from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, DepthwiseConv2D, Add
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

def image_gen_w_aug(train_parent_directory):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        shear_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        validation_split=0.15  # 15% for validation
    )

    train_generator = train_datagen.flow_from_directory(
        train_parent_directory,
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_generator = train_datagen.flow_from_directory(
        train_parent_directory,
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    return train_generator, val_generator


def cnn_architecture(input_shape=(128, 128, 3)):
    inputs = Input(shape=input_shape)

    # Block 1
    x = Conv2D(32, (3,3), padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3,3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = Dropout(0.25)(x)

    # Block 2
    x = Conv2D(64, (3,3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3,3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = Dropout(0.25)(x)

    # Block 3
    x = Conv2D(128, (3,3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3,3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = Dropout(0.3)(x)

    # Head
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs)

<<<<<<< HEAD
train_dir = 'C:\Python\group 5 mlai pancakes\train'
test_dir = 'C:\Python\group 5 mlai pancakes\test'
=======
train_dir = 'C:/Python/rps/datasets/train/'
test_dir = 'C:/Python/rps/datasets/test/'
>>>>>>> 2dde8a9 (Initial commit for Htoo branch)

# Load data with 15% validation split
train_generator, validation_generator = image_gen_w_aug(train_dir)

print("Class indices (label mapping):", train_generator.class_indices)

# Test data remains the same
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)


# Compute class weights
classes = np.unique(train_generator.classes)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_generator.classes)
class_weights = dict(zip(classes, weights))

# Build and compile model
model_TL = cnn_architecture()
model_TL.compile(optimizer=Adam(learning_rate=1e-5),
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train model
history_TL = model_TL.fit(train_generator,
                          steps_per_epoch=len(train_generator),
                          epochs=35,
                          validation_data=validation_generator,
                          validation_steps=len(validation_generator),
                          class_weight=class_weights,
                          callbacks=[early_stop],
                          verbose=1)

# Plot accuracy
plt.plot(history_TL.history['accuracy'], label='train_acc')
plt.plot(history_TL.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()

# Save model
model_TL.summary()
<<<<<<< HEAD
model_TL.save('model_pluto.keras')
=======
model_TL.save('model_pluto.keras')
>>>>>>> 2dde8a9 (Initial commit for Htoo branch)
