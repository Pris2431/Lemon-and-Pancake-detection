import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

# Function to preprocess and predict
def import_and_predict(image_data, model):
    size = (128, 128)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    image = image.convert('RGB')
    image = np.asarray(image)
    image = (image.astype(np.float32) / 255.0)
    img_reshape = image[np.newaxis, ...]
    prediction = model.predict(img_reshape, verbose=0)
    return prediction

# Load trained model
model = tf.keras.models.load_model('C:/Python/rps/model_pluto.keras')  # or model_TL.keras

# App title
st.title("Lemon vs Pancake Classifier")
st.write("Upload an image, and I’ll tell you if it’s a lemon, a pancake, or something unknown!")

# File uploader
file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if file is None:
    st.info("📷 Please upload an image file.")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True, caption="Uploaded Image")

    prediction = import_and_predict(image, model)

    class_names = ['lemon', 'pancake', 'unknown']
    confidence = np.max(prediction)
    predicted_index = np.argmax(prediction)

    if confidence < 0.4:
        result = "It is unknown."
    else:
        result = f"It is a {class_names[predicted_index]} ✅"

    # Display prediction
    st.markdown(f"### 🔍 {result}")
    st.markdown(f"**Confidence:** {confidence:.2f}")

    # Display all class probabilities
    st.subheader("Class Probabilities:")
    for name, prob in zip(class_names, prediction[0]):
        st.write(f"- **{name}**: {prob:.4f}")
