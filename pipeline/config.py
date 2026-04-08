import os

# Rutas principales del Dataset
BASE_DIR = r"d:\LSC"

# LSC-54 (LSCS45)
LSC54_DIR = os.path.join(BASE_DIR, "LSCS45")
LSC54_JSON_PATH = os.path.join(LSC54_DIR, "datos.json") # Archivo pesado
LSC54_SAMPLE_JSON_PATH = os.path.join(LSC54_DIR, "sample.json")

# LSC50
LSC50_DIR = os.path.join(BASE_DIR, "LSC50")
LSC50_LANDMARKS_DIR = os.path.join(LSC50_DIR, "LANDMARKS")

# LSC70
LSC70_DIR = os.path.join(BASE_DIR, "LSC70", "LSC70")

# LSCPROPIO (Videos propios .mp4/.m4v)
LSCPROPIO_DIR = os.path.join(BASE_DIR, "LSCPROPIO")

# Rutas de salida del Pipeline
OUTPUT_DIR = os.path.join(BASE_DIR, "pipeline_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Formatos de columnas padronizadas para Landmarks (Pose, Hands, Face)
# MediaPipe Holistic genera hasta:
# - 33 pose landmarks
# - 21 left hand landmarks
# - 21 right hand landmarks
# - 468 face landmarks (opcional, en LSC50 pueden diferir)
