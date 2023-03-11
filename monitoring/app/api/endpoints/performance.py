"""Endpoint para cálculo de Performance."""
from fastapi import APIRouter
import pandas as pd
import pickle
from sklearn.metrics import roc_auc_score
import os
import numpy as np

router = APIRouter(prefix="/performance")

@router.post("/")
async def performance(data: list[dict]):
    # Carregando o modelo pré-treinado
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../model.pkl'))
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Transformando a lista de registros em um DataFrame
    df = pd.DataFrame(data).fillna(np.nan)

    # Separando o mês da data de referência
    df["month"] = pd.to_datetime(df["REF_DATE"]).dt.month

    # Calculando a volumetria para cada mês
    volumes = df["month"].value_counts().to_dict()

    # Escorando o DataFrame com o modelo pré-treinado
    df["score"] = model.predict_proba(df.drop(["REF_DATE", "TARGET", "month"], axis=1))[:, 1]
    

    # Calculando a performance do modelo
    roc_auc = roc_auc_score(df["TARGET"], df["score"])

    # Retornando a volumetria e a performance
    return {
        "volumes": volumes,
        "roc_auc": roc_auc
    }