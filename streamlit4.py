import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('custom_cnn_v4_pancake_lemon_nothing.h5')
class_names = ['pancake', 'lemon', 'nothing']

st.title("Pancake, Lemon, or Nothing Classifier (Custom CNN v4)")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    image = ImageOps.fit(image, (128, 128), Image.Resampling.LANCZOS)
    img = np.asarray(image).astype(np.float32) / 255.0
    img = img[np.newaxis, ...]
    prediction = model.predict(img)
    st.write(f"Prediction: {class_names[np.argmax(prediction)]}")
    st.write(f"Probabilities: {prediction[0]}")
