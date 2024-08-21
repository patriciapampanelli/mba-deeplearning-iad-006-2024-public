import os
import base64
from datetime import datetime
from PIL import Image, ImageOps
import cv2
from io import BytesIO
import numpy as np
import pickle as pk

class PredictDigit():
    NORMALIZER = pk.load(open('./models/normalizer.sav', 'rb'))
    MODELS = {
        'decision_tree': pk.load(open('./models/decision_tree.sav', 'rb')),
        'random_forest': pk.load(open('./models/random_forest.sav', 'rb')),
        'xgboost': pk.load(open('./models/xgboost.sav', 'rb'))
    }
    DEBUG_INFERENCE = os.environ.get("DEBUG_INFERENCE", False)
    DEBUG_DIGIT_IMAGES = os.environ.get("DEBUG_DIGIT_IMAGES", False)

    def __init__(self, image, model_name):
        self.image = image
        self.model_name = model_name

    def call(self):
        image_data = self.__parse_image()
        probabilities = PredictDigit.MODELS[self.model_name].predict_proba(
            PredictDigit.NORMALIZER.transform(image_data)
        )
        _, result = np.unravel_index(probabilities.argmax(), probabilities.shape)

        if PredictDigit.DEBUG_INFERENCE:
            print(f'Image data was {image_data}')
            print(f'Predicted digit was {result}')
            print(f'Model used for the inference was {self.model_name}')
            print(f'The array of probabilities was {probabilities}')

        return { 'digit': int(result), 'probability': "{:.2f}".format(float(probabilities[0][result])), 'model': self.model_name }

    def __parse_image(self):
        if "data:image" in self.image:
            self.image = self.image.split(",")[1]

        image_arr = base64.b64decode(self.image)
        nparr = np.fromstring(image_arr, np.uint8)
        image = cv2.imdecode(nparr, flags=cv2.IMREAD_COLOR)
        gray_scale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(gray_scale_image, (8, 8))

        if PredictDigit.DEBUG_DIGIT_IMAGES:
            now = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
            cv2.imwrite('./user_images/number_'+ now +'.png', image)
            cv2.imwrite('./user_images/number_gray_scale_'+ now +'.jpg', gray_scale_image)
            cv2.imwrite('./user_images/resized_number_gray_scale_'+ now +'.jpg', resized_image)

        original_image_data = np.asarray(resized_image).flatten().reshape(1, -1)[0]
        pixel_max_value = max(original_image_data)
        image_data = [(original_image_data * (16 / pixel_max_value)).astype(int)]
        return image_data
