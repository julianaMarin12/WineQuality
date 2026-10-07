"""
Regenera los artefactos del modelo (knn_imputer, iqr_bounds, standard_scaler,
best_svm_model) a partir de la base procesada del notebook.

Uso:
    python train_model.py                       # busca data/base procesada.csv
    python train_model.py ruta/a/mi_base.csv
"""
import sys
from pathlib import Path

import pandas as pd

from pipeline import MODELS_DIR, find_training_csv, save_artifacts, train_artifacts

csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else find_training_csv()
if csv_path is None or not csv_path.exists():
    sys.exit("No se encontró la base. Copia 'base procesada.csv' en la carpeta data/.")

df = pd.read_csv(csv_path)
art = train_artifacts(df)
save_artifacts(art)

print(f"Base usada: {csv_path} ({len(df)} registros)")
print(f"Artefactos guardados en: {MODELS_DIR}/")
