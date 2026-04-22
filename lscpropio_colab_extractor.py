# ==============================================================================
# LSCPROPIO EXTRACTOR PARA GOOGLE COLAB
# Instrucciones de uso en Google Colab:
# 1. Abre un nuevo entorno en https://colab.research.google.com/
# 2. Sube la carpeta "LSCPROPIO" al almacenamiento de la izquierda (el icono de la carpeta).
# 3. Sube este script "lscpropio_colab_extractor.py".
# 4. En una celda de código, instala MediaPipe:
#      !pip install mediapipe opencv-python pandas tqdm
# 5. Ejecuta el script:
#      !python lscpropio_colab_extractor.py
# 6. Descarga el archivo "LSCPROPIO_CSVs.zip" resultante y pégalo localmente
#    en d:\LSC\pipeline_output\dynamic_landmarks\
# ==============================================================================

import os
import glob
import zipfile
import pandas as pd
import numpy as np
import cv2
import mediapipe.python.solutions.holistic as mp_holistic
from tqdm import tqdm

NUM_POSE_LANDMARKS = 33
NUM_HAND_LANDMARKS = 21
NUM_FACE_LANDMARKS = 468

def process_video(video_path, output_dir):
    """Procesa un video y guarda un archivo .csv con 1629 columnas."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False

    records = []
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret: break

            results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            row = {}

            # -- POSE (99 cols) --
            if results.pose_landmarks:
                for i, lm in enumerate(results.pose_landmarks.landmark):
                    row[f'pose_{i}_x'] = lm.x
                    row[f'pose_{i}_y'] = lm.y
                    row[f'pose_{i}_z'] = lm.z
            else:
                for i in range(NUM_POSE_LANDMARKS):
                    row[f'pose_{i}_x'] = np.nan
                    row[f'pose_{i}_y'] = np.nan
                    row[f'pose_{i}_z'] = np.nan
                    
            # -- L_HAND (63 cols) --
            if results.left_hand_landmarks:
                for i, lm in enumerate(results.left_hand_landmarks.landmark):
                    row[f'l_hand_{i}_x'] = lm.x
                    row[f'l_hand_{i}_y'] = lm.y
                    row[f'l_hand_{i}_z'] = lm.z
            else:
                for i in range(NUM_HAND_LANDMARKS):
                    row[f'l_hand_{i}_x'] = np.nan
                    row[f'l_hand_{i}_y'] = np.nan
                    row[f'l_hand_{i}_z'] = np.nan

            # -- R_HAND (63 cols) --
            if results.right_hand_landmarks:
                for i, lm in enumerate(results.right_hand_landmarks.landmark):
                    row[f'r_hand_{i}_x'] = lm.x
                    row[f'r_hand_{i}_y'] = lm.y
                    row[f'r_hand_{i}_z'] = lm.z
            else:
                for i in range(NUM_HAND_LANDMARKS):
                    row[f'r_hand_{i}_x'] = np.nan
                    row[f'r_hand_{i}_y'] = np.nan
                    row[f'r_hand_{i}_z'] = np.nan

            # -- FACE (1404 cols) --
            if results.face_landmarks:
                for i, lm in enumerate(results.face_landmarks.landmark):
                    row[f'face_{i}_x'] = lm.x
                    row[f'face_{i}_y'] = lm.y
                    row[f'face_{i}_z'] = lm.z
            else:
                for i in range(NUM_FACE_LANDMARKS):
                    row[f'face_{i}_x'] = np.nan
                    row[f'face_{i}_y'] = np.nan
                    row[f'face_{i}_z'] = np.nan

            records.append(row)
            
    cap.release()
    
    if not records:
        return False

    # Extraemos el nombre correcto (LSCPROPIO/celular/celular-persona1.mp4 -> CELULAR)
    # Algunos videos están en subcarpetas, extraemos el nombre de su carpeta padre.
    parent_dir = os.path.basename(os.path.dirname(video_path))
    if parent_dir.lower() == "lscpropio":
        label = os.path.splitext(os.path.basename(video_path))[0].split("-")[0].upper()
    else:
        label = parent_dir.upper()
        
    df = pd.DataFrame(records)
    # Llenamos los NaN con 0.0 que es lo que espera el engine IK
    df.fillna(0.0, inplace=True)
    
    csv_path = os.path.join(output_dir, f"{label}.csv")
    
    # IMPORTANTE: Si ya existe un CSV (porque hay varios videos de la misma seña), omitimos 
    # para usar el primero detectado, o se puede promediar/unir. Aquí guardamos 1 solo.
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=True)
        return True
    return False

def main():
    # Soportar Docker Environments / Colab via Variables de Entorno
    source_dir = os.getenv("LSC_INPUT", "LSCPROPIO")
    out_dir = os.getenv("LSC_OUTPUT", "extracted_csvs")
    os.makedirs(out_dir, exist_ok=True)
    
    videos = glob.glob(f"{source_dir}/**/*.mp4", recursive=True)
    if not videos:
        print("No se encontraron videos MP4 en la carpeta LSCPROPIO/")
        return
        
    print(f"Encontrados {len(videos)} videos. Iniciando extraccion MediaPipe...")
    success_count = 0
    for v in tqdm(videos):
        if process_video(v, out_dir):
            success_count += 1
            
    print(f"Extraccion finalizada. CSVs procesados: {success_count}.")
    
    # Zippear la salida
    zip_filename = "LSCPROPIO_CSVs.zip"
    print(f"Empaquetando en {zip_filename}...")
    with zipfile.ZipFile(zip_filename, 'w') as zf:
        for file in glob.glob(f"{out_dir}/*.csv"):
            zf.write(file, os.path.basename(file))
            
    print("¡Finalizado! Por favor descarga el archivo LSCPROPIO_CSVs.zip.")

if __name__ == "__main__":
    main()
