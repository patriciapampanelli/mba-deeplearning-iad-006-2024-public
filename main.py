from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Union
from numpy import ndarray
from pydantic import BaseModel
from enum import Enum
import os
from PIL import Image
import io
import base64

app = FastAPI(
    title="API do exercício de IAD-014: ML2 - Árvores de Decisão",
    description="Esta é uma API usando FastAPI e Swagger UI que recebe uma imagem de entrada e faz a previsão do número baseado em um modelo de árvore de decisão.",
    version="1.0.0",
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
)

class Item(BaseModel):
    name: str
    price: float
    is_offer: Union[bool, None] = None

class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"

#classe para importacao da imagem
class ImageData(BaseModel):
    filename: str
    content: str  # Base64 encoded image content

@app.get("/")
def hello_world():
    return {"message": "Olá, esta é a API do exercício de IAD-014: ML2 - Árvores de Decisão!"}

@app.post("/predict/")
def predict_image(image: ImageData):
    # Decode the base64 image content
    image_content = base64.b64decode(image.content)
    
    # Process the image
    image = Image.open(io.BytesIO(image_content))
    return {"image": image}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item_name": item.name, "item_id": item_id}

@app.get("/models/{model_name}")
async def get_model(model_name: ModelName):
    if model_name is ModelName.alexnet:
        return {"model_name": model_name, "message": "Deep Learning FTW!"}

    if model_name.value == "lenet":
        return {"model_name": model_name, "message": "LeCNN all the images"}

    return {"model_name": model_name, "message": "Have some residuals"}

@app.get("/file/{file_path:path}")
async def read_file(file_path: str):
    if os.path.isfile(file_path):
        return {"file_path": f"{file_path} encontrado com sucesso"}
    else:
        return {"file_path": f"{file_path} não encontrado"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

