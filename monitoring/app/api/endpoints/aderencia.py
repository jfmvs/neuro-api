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
async def aderencia(file_path: list[str]):
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../model.pkl'))

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Lendo a base de dados a ser avaliada
    with gzip.open(file_path[0], "rb") as f:
        new_df = pd.read_csv(f)

    for feature in new_df.columns:
        if not pd.api.types.is_numeric_dtype(new_df[feature]):
            categories = model.classes_ if hasattr(model, 'classes_') else new_df[feature].unique()
            
            for i, value in new_df[feature].items():
                if value not in categories:
                    new_df.at[i, feature] = float('nan')

    # Escorando a base de dados com o modelo pré-treinado
    new_df["score"] = model.predict_proba(new_df)[:, 1]

    # Lendo a base de teste
    test_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')), 'datasets', 'credit_01', 'test.gz')
    with gzip.open(test_path, "rb") as g:
        test_df = pd.read_csv(g)

    # Calculando a métrica de Kolmogorov-Smirnov entre as distribuições de score da base fornecida e da base de teste
    ks_statistic, p_value = ks_2samp(new_df["score"], test_df["TARGET"])

    # Retornando a métrica calculada
    return {
        "KS": ks_statistic,
        "p-value": p_value
    }