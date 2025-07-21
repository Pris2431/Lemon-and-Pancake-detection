from PIL import Image, ImageOps
import tensorflow as tf
import cv2
import numpy as np
import sys

# Load model
model = tf.keras.models.load_model('C:/Python/rps/model_pluto.keras')
print("Model loaded!")

# These must match the order used during training
class_names = ['lemon', 'pancake', 'unknown']

def import_and_predict(image_data, model):
    size = (128, 128)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    image = image.convert('RGB')
    image = np.asarray(image)
    image = image.astype(np.float32) / 255.0
    img_reshape = image[np.newaxis, ...]
    prediction = model.predict(img_reshape, verbose=0)
    return prediction

cap = cv2.VideoCapture(0)

if cap.isOpened():
    print("Camera OK")
else:
    print("Failed to open camera.")
    sys.exit()

while True:
    ret, original = cap.read()
    if not ret:
        continue

    # Optional: Improve lighting
    original = cv2.convertScaleAbs(original, alpha=1.3, beta=40)

    frame = cv2.resize(original, (128, 128))
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    prediction = import_and_predict(image, model)
    confidence = np.max(prediction)
    class_index = np.argmax(prediction)

    if confidence < 0.6:
        label_text = "It is unknown"
    else:
        detected = class_names[class_index]
        label_text = f"It is a {detected}"

    cv2.putText(original, label_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Webcam Classification", original)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
