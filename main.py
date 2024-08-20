from fastapi import FastAPI, File
from pydantic import BaseModel
import xgboost as xgb
import numpy as np
import pickle
import warnings

import base64
from PIL import Image
import io

warnings.simplefilter(action='ignore', category=DeprecationWarning)

app = FastAPI()

#Definição dos tipos de dados
class PredictionResponse(BaseModel):
  prediction: float

class Imagerequest(BaseModel):
  image: str
