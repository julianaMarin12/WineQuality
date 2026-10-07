"""
Pipeline de inferencia del modelo de calidad de vino.

Replica exactamente el flujo del notebook "Trabajo Final - Wine Quality Modelamiento":
  1. Ceros inválidos y valores fuera de límites físicos -> NaN
  2. Imputación con KNNImputer (k=5)
  3. Capping IQR (Q1 - 1.5·IQR, Q3 + 1.5·IQR)
  4. StandardScaler
  5. SVR optimizado (kernel='linear', C=1.0, epsilon=0.01)
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATA_CANDIDATES = [
    BASE_DIR / "data" / "base procesada.csv",
    BASE_DIR / "data" / "base_procesada.csv",
    BASE_DIR / "base procesada.csv",
    BASE_DIR / "base_procesada.csv",
]

TARGET = "quality"
FEATURES = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "density",
    "pH",
    "alcohol",
]

# Límites físicos usados en el ETL del notebook
LIMITS = {
    "fixed_acidity": (4.0, 8.5),
    "volatile_acidity": (0.0, 1.2),
    "citric_acid": (0.0, 1.0),
    "residual_sugar": (0.0, 45.0),
    "chlorides": (0.0, 0.6),
    "free_sulfur_dioxide": (0.0, 50.0),
    "density": (0.98, 1.04),
    "pH": (2.8, 4.0),
    "alcohol": (7.0, 16.0),
}
ZERO_IS_MISSING = ["pH", "density", "alcohol", "fixed_acidity"]

# Columnas que el notebook descarta si vienen en el archivo
DROP_COLS = [
    "RecordID", "FullName", "Phone", "ZodiacSign", "FavoriteColor", "Hobby",
    "type", "sulphates", "total_sulfur_dioxide",
]

ARTIFACT_FILES = {
    "imputer": "knn_imputer.joblib",
    "iqr": "iqr_bounds.joblib",
    "scaler": "standard_scaler.joblib",
    "model": "best_svm_model.joblib",
}


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina columnas irrelevantes y marca como NaN los valores imposibles."""
    out = df.drop(columns=[c for c in DROP_COLS if c in df.columns]).copy()
    for col in ZERO_IS_MISSING:
        if col in out.columns:
            out[col] = out[col].replace(0, np.nan)
    for col, (lo, hi) in LIMITS.items():
        if col in out.columns:
            out.loc[(out[col] < lo) | (out[col] > hi), col] = np.nan
    return out


def train_artifacts(df_raw: pd.DataFrame) -> dict:
    """Entrena imputador, límites IQR, escalador y SVR igual que en el notebook."""
    df = clean_raw(df_raw)
    df = df[FEATURES + [TARGET]].astype(float)
    df = df.dropna(subset=[TARGET])

    imputer = KNNImputer(n_neighbors=5)
    imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    iqr_bounds = {}
    for col in FEATURES:
        q1, q3 = imputed[col].quantile(0.25), imputed[col].quantile(0.75)
        iqr = q3 - q1
        iqr_bounds[col] = {"lower": q1 - 1.5 * iqr, "upper": q3 + 1.5 * iqr}
        imputed[col] = np.clip(imputed[col], iqr_bounds[col]["lower"], iqr_bounds[col]["upper"])

    X, y = imputed[FEATURES], imputed[TARGET]
    scaler = StandardScaler().fit(X)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=FEATURES)

    model = SVR(kernel="linear", C=1.0, epsilon=0.01).fit(X_scaled, y)

    return {
        "imputer": imputer,
        "iqr": iqr_bounds,
        "scaler": scaler,
        "model": model,
        "train_stats": X.describe().T,
        "y_range": (float(y.min()), float(y.max())),
        "y_mean": float(y.mean()),
    }


def save_artifacts(art: dict, folder: Path = MODELS_DIR) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for key, fname in ARTIFACT_FILES.items():
        joblib.dump(art[key], folder / fname)


def load_artifacts(folder: Path = MODELS_DIR) -> dict:
    return {key: joblib.load(folder / fname) for key, fname in ARTIFACT_FILES.items()}


def find_training_csv() -> Path | None:
    return next((p for p in DATA_CANDIDATES if p.exists()), None)


def preprocess(df_in: pd.DataFrame, art: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (valores limpios en unidades reales, valores escalados para el modelo)."""
    df = clean_raw(df_in)
    for col in FEATURES:
        if col not in df.columns:
            df[col] = np.nan
    df = df[FEATURES].astype(float)

    # El imputador del notebook se ajustó con 'quality' incluida: se agrega como NaN
    imputer = art["imputer"]
    cols_fit = list(getattr(imputer, "feature_names_in_", FEATURES + [TARGET]))
    tmp = df.copy()
    for col in cols_fit:
        if col not in tmp.columns:
            tmp[col] = np.nan
    tmp = tmp[cols_fit]
    imputed = pd.DataFrame(imputer.transform(tmp), columns=cols_fit)[FEATURES]

    capped = imputed.copy()
    for col, b in art["iqr"].items():
        if col in capped.columns:
            capped[col] = np.clip(capped[col], b["lower"], b["upper"])

    scaled = pd.DataFrame(art["scaler"].transform(capped[FEATURES]), columns=FEATURES)
    return capped, scaled


def predict(df_in: pd.DataFrame, art: dict) -> np.ndarray:
    _, scaled = preprocess(df_in, art)
    return art["model"].predict(scaled)


def contributions(df_row: pd.DataFrame, art: dict) -> pd.Series:
    """Aporte de cada variable a la predicción (válido porque el kernel es lineal)."""
    _, scaled = preprocess(df_row, art)
    coef = np.ravel(art["model"].coef_)
    return pd.Series(coef * scaled.iloc[0].values, index=FEATURES)
