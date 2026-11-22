# Capítulo 14 — MLOps, implantação e projeto integrador
# Bloco 3. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
# ATENÇÃO: este trecho NÃO foi executado na elaboração do livro. Fixe versões e confira a documentação.
from fastapi import FastAPI
from pydantic import BaseModel
import joblib, numpy as np

app = FastAPI()
modelo = joblib.load("modelo.joblib")           # carregado uma vez, na partida do serviço

class Pedido(BaseModel):
    atributos: list[float]                      # os 30 atributos, na mesma ordem do treino

@app.post("/prever")
def prever(p: Pedido):
    x = np.array(p.atributos).reshape(1, -1)
    return {"classe": int(modelo.predict(x)[0]), "probabilidade": float(modelo.predict_proba(x)[0].max())}
