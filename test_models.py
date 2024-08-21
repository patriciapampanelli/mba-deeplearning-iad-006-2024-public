import numpy as np
import cv2
import base64
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from predict_digit import PredictDigit

def load_image(path_with_name):
    image = cv2.imread(path_with_name, cv2.IMREAD_GRAYSCALE)
    #equalized_img = cv2.equalizeHist(image)
    #equalized_img = equalized_img.reshape(1, -1)
    #normalized_img = equalized_img[0] / 255
    return image

def base64_to_image(base64_str):
    png_recovered = base64.b64decode(base64_str)
    with open('./Dataset/captured_2.jpeg', "wb") as f:
        f.write(png_recovered)

def image_file_to_base64(image_file_path):
    with open(image_file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string

def image_array_to_base64(image):
    processed_string = base64.b64encode(image.tobytes()).decode()
    return processed_string

img = image_file_to_base64('./user_images/number_08_13_2024_19_55_55.png')
print(img)
digits = load_digits()
X = digits.data
Y = digits.target
#print(X[1])
#print(Y[1])
img_2 = image_array_to_base64(X[1])
print(img_2)
PredictDigit(img_2, 'decision_tree').call()
