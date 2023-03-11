"""Endpoint para cálculo de aderência."""
from fastapi import APIRouter
import os
import gzip
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import pickle

router = APIRouter(prefix="/aderencia")

@router.post("/")
async def aderencia(file_path: str):
    print(file_path)
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../model.pkl'))
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Lendo a base de dados a ser avaliada
    with gzip.open(file_path, "rb") as f:
        new_df = pd.read_csv(f)

    # Escorando a base de dados com o modelo pré-treinado
    new_df["score"] = model.predict_proba(new_df.drop(["REF_DATE"], axis=1))[:, 1]

    # Lendo a base de teste anterior
    test_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../batch_records.json'))
    test_df = pd.read_json(test_path)

    # Calculando a métrica de Kolmogorov-Smirnov entre as distribuições de score da base fornecida e da base de teste
    ks_statistic, p_value = ks_2samp(new_df["score"], test_df["TARGET"])

    # Retornando a métrica calculada
    return {
        "KS": ks_statistic,
        "p-value": p_value
    }
