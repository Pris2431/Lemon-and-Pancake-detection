import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, callbacks

train_dir = 'C:/Python/group 5 mlai pancakes/train/'
valid_dir = 'C:/Python/group 5 mlai pancakes/valid/'
test_dir  = 'C:/Python/group 5 mlai pancakes/test/'

IMG_SIZE = (128, 128)   # Reduced size for speed
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
valid_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
)
valid_gen = valid_datagen.flow_from_directory(
    valid_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
)

# --- EfficientNetB0, streamlined top ---
base_model = tf.keras.applications.EfficientNetB0(
    include_top=False, weights='imagenet', input_shape=(128, 128, 3)
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(128, 128, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(64, activation='relu')(x)  # Fewer neurons
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(3, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- Early Stopping ---
es = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

EPOCHS = 10  # Can increase if you have time
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=valid_gen,
    callbacks=[es]
)

#for more accuracy
base_model.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, epochs=5, validation_data=valid_gen, callbacks=[es])

# ---- Save model ----
model.save('efficientnetb0_v3.h5')
print("Model saved as efficientnetb0_v3.h5")
