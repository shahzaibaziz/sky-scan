import cv2
import numpy as np
from tensorflow.keras.models import load_model

def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (256, 256))
    image = np.expand_dims(image, axis=0)
    return image

def predict(image_path, model_path):
    model = load_model(model_path)
    image = load_image(image_path)
    prediction = model.predict(image)
    return prediction

if __name__ == "__main__":
    image_path = 'data/raw/sample_image.jpg'
    model_path = 'models/satellite_model.h5'
    prediction = predict(image_path, model_path)
    print(prediction)