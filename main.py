import pickle
from PIL import Image

import numpy as np
import xgboost as xgb
from fastapi import FastAPI
from fastapi import File
from fastapi import UploadFile
from fastapi.responses import JSONResponse
from fastapi.responses import RedirectResponse


app = FastAPI()
model_path = "notebooks/xgb_clf.pkl"


with open(model_path, "rb") as file:
    model = pickle.load(file)


def preprocess_image(image):
    image = image.resize((8, 8))
    image = image.convert('L')
    image_array = np.array(image)
    image_array = image_array / 16.0
    image_array = image_array.flatten()
    return image_array


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file)
        image_processed = preprocess_image(image)
        dmatrix = xgb.DMatrix([image_processed])
        prediction = model.predict(dmatrix)
        return JSONResponse(content={"prediction": int(prediction[0])})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/health-check")
async def app_check():
    return {"message": "OK"}


@app.get("/")
async def root():
    return RedirectResponse(url="/docs")
