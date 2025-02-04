import numpy as np
import cv2

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    # Add preprocessing steps here
    return image