#  Predictor de Calidad del Vino (SVR optimizado)

**Integrantes:** Maria Camila Duque y Carmen Juliana Marin

App de Streamlit construida a partir del notebook *Trabajo Final – Wine Quality Modelamiento*.
Usa el modelo ganador del estudio: **SVR con kernel lineal (C=1.0, ε=0.01)**, con el mismo
pipeline de preprocesamiento (KNNImputer → capping IQR → StandardScaler).

## Estructura

```
├── app.py                  # La app
├── pipeline.py             # Preprocesamiento + predicción (idéntico al notebook)
├── train_model.py          # Regenera los .joblib desde la base procesada
├── requirements.txt
├── .streamlit/config.toml  # Tema vino
├── models/                 # Aquí van los 4 archivos .joblib
└── data/
    ├── ejemplo_lote.csv    # Ejemplo para probar la pestaña de lotes
    └── base procesada.csv  # (opcional) la base usada en el notebook
```

## 1. Pon el modelo en la carpeta `models/`

Descarga desde Colab los archivos que generó el notebook y cópialos en `models/`:

- `knn_imputer.joblib`
- `iqr_bounds.joblib`
- `standard_scaler.joblib`
- `best_svm_model.joblib`

En Colab, revisa la versión de scikit-learn y fíjala en `requirements.txt`
(por ejemplo `scikit-learn==1.6.1`), porque los `.joblib` solo cargan bien con la misma versión:

```python
import sklearn; print(sklearn.__version__)
```

## 2. Probar en tu computador

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Qué hace la app

- **Analizar un vino:** sliders agrupados por acidez, azúcar y sales, y cuerpo; perfiles de ejemplo;
  copa que se llena según la calidad estimada y gráfico con el aporte de cada variable.
- **Analizar un lote (CSV):** predicción masiva, MAE si el archivo trae `quality`,
  gráfico real vs. predicho y descarga de resultados.
- **Cómo funciona el modelo:** pipeline, comparación de los 8 modelos (5-Fold CV) y pesos del SVR.
