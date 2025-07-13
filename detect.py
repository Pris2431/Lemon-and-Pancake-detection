from PIL import Image, ImageOps
import tensorflow as tf
import numpy as np
import cv2
import os

# Load your custom model
model = tf.keras.models.load_model('v1_model.hdf5')
class_names = ['pancake', 'lemon', 'nothing']

def import_and_predict(image_data, model):
    size = (75,75)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    image = image.convert('RGB')
    image = np.asarray(image).astype(np.float32) / 255.0
    img_reshape = image[np.newaxis,...]
    prediction = model.predict(img_reshape)
    return prediction

cap = cv2.VideoCapture(0)
if (cap.isOpened()):
    print("Camera OK")
else:
    cap.open()

while (True):
    ret, original = cap.read()
    frame = cv2.resize(original, (224, 224))
    cv2.imwrite(filename='img.jpg', img=original)
    image = Image.open('img.jpg')
    prediction = import_and_predict(image, model)
    pred_class = class_names[np.argmax(prediction)]
    predict = f"It is a {pred_class}!"

    cv2.putText(original, predict, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow("Classification", original)

    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
