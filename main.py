from fastapi import FastAPI
import uvicorn
import numpy as np
import matplotlib.pyplot as plt
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from typing import List
import pickle as pkl
import warnings

warnings.simplefilter(action='ignore', category=DeprecationWarning)	

# Cria uma instância do FastAPI
app = FastAPI(
    title="API do exercício de IAD-014: ML2 - Árvores de Decisão",
    description="Esta é uma API usando FastAPI e Swagger UI que recebe uma imagem de entrada e faz a previsão do número baseado em um modelo de árvore de decisão.",
    version="1.0.0",
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# Define o modelo de dados (lista de inteiros) que será recebido no corpo da requisição POST
class ImageData(BaseModel):
    image: List[List[int]]

#Implementa a rota raiz que exibe mensagem de utilizacao da API
@app.get("/")
def hello():
    html_content = """
    <html>
        <head>
            <title>Bem-vindo à API de Árvores de Decisão</title>
        </head>
        <body>
            <h1>Olá, esta é a API do exercício de IAD-014: ML2 - Árvores de Decisão!</h1>
            <p>Para utilizar, envie na rota <strong>/predict/</strong> uma lista com 64 inteiros (0 a 16) que representam os pixels de uma imagem 8x8.</p>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# Inicialização da Aplicação	
@app.on_event("startup")	
async def startup_event():	
    load_model()	

def load_model():
    global clf
    try:
        file_path = './models/modelo.pkl'  
        with open(file_path, 'rb') as f:
            # Pickle the clf model using the highest protocol available.
            clf = pkl.load(f)
    except FileNotFoundError:
        print(f"Erro interno: Não foi possível carregar o modelo em {file_path}")

#Implementa a rota POST no caminho "/predict/" que aceita dados no formato definido por ImageData.
@app.post("/predict/")
async def predict(data: ImageData):
    # Converte a lista recebida de volta para um array numpy com tipo uint8 
    image = np.array(data.image, dtype=np.uint8)
    y_pred = clf.predict(image)

    #retorna o resultado da predicao e dados do modelo
    data = {
        "data": image.tolist(),
        "predict": y_pred.tolist(),
        "description": "Resultado da previsao.",
        "details": f"Profundidade {clf.tree_.max_depth}"
    }
    return JSONResponse(content=data)

#Implementa a rota "/healthcheck/" que retorna "ok" para sinalizar que o servidor está oline.
@app.get("/healthcheck")
async def healthcheck():
    #retorna ok
    data = {
        "status": "ok",
    }
    return JSONResponse(content=data)

# Se o arquivo for executado diretamente, inicia o servidor uvicorn no endereço 0.0.0.0 e porta 8000.
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
