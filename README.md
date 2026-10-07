# 🍷 Predictor de Calidad del Vino (SVR optimizado)

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

Opción A (recomendada): descarga desde Colab los archivos que generó el notebook y cópialos en `models/`:

- `knn_imputer.joblib`
- `iqr_bounds.joblib`
- `standard_scaler.joblib`
- `best_svm_model.joblib`

En Colab, revisa la versión de scikit-learn y fíjala en `requirements.txt`
(por ejemplo `scikit-learn==1.6.1`), porque los `.joblib` solo cargan bien con la misma versión:

```python
import sklearn; print(sklearn.__version__)
```

Opción B: copia `base procesada.csv` en `data/`. Si la app no encuentra los `.joblib`
(o no puede cargarlos), entrena el modelo automáticamente al iniciar con esa base.
También puedes generarlos localmente con `python train_model.py`.

## 2. Probar en tu computador

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 3. Desplegar en Streamlit Community Cloud

1. Sube esta carpeta a un repositorio de GitHub (incluyendo `models/` o `data/base procesada.csv`).
2. Entra a https://share.streamlit.io y elige **Create app**.
3. Selecciona el repositorio, la rama y `app.py` como archivo principal.
4. En *Advanced settings* elige la misma versión de Python que usaste en Colab si fijaste scikit-learn.
5. Pulsa **Deploy**.

## Qué hace la app

- **Analizar un vino:** sliders agrupados por acidez, azúcar y sales, y cuerpo; perfiles de ejemplo;
  copa que se llena según la calidad estimada y gráfico con el aporte de cada variable.
- **Analizar un lote (CSV):** predicción masiva, MAE si el archivo trae `quality`,
  gráfico real vs. predicho y descarga de resultados.
- **Cómo funciona el modelo:** pipeline, comparación de los 8 modelos (5-Fold CV) y pesos del SVR.
