import os
import json
import numpy as np
import config

def export_dummy_web_json():
    print("Extrayendo secuencia de la Matriz NumPy para el reproductor WebGL...")
    
    tensor_path = os.path.join(config.OUTPUT_DIR, "LSC_dataset_unificado.npy")
    if not os.path.exists(tensor_path):
        print("Error: No se encontró el dataset unificado.npy")
        return
        
    X = np.load(tensor_path) # Shape: (samples, frames=60, features=1629)
    
    # Tomamos la primer muestra de entrenamiento (una seña completa de 60 frames)
    sample_seq = X[0]
    
    frames_json = []
    
    # Desempacar las 1629 features matemáticamente (33 pose + 21 lh + 21 rh + 468 face) * 3 xyz
    for frame_idx in range(len(sample_seq)):
        features = sample_seq[frame_idx]
        cursor = 0
        
        # Helper para extraer arrays de diccionarios {x, y, z}
        def get_landmarks(num_points):
            nonlocal cursor
            pts = []
            for _ in range(num_points):
                # Para evitar problemas con Kalidokit en web, clamp values y mapea a float standard
                pts.append({
                    "x": float(features[cursor]),
                    "y": float(features[cursor+1]),
                    "z": float(features[cursor+2])
                })
                cursor += 3
            return pts

        pose_lm = get_landmarks(33)
        lh_lm = get_landmarks(21)
        rh_lm = get_landmarks(21)
        face_lm = get_landmarks(468)
        
        frames_json.append({
            "poseLandmarks": pose_lm,
            "pose3DLandmarks": pose_lm, # El MediaPipe local nos dio 3D absoluto
            "leftHandLandmarks": lh_lm,
            "rightHandLandmarks": rh_lm,
            "faceLandmarks": face_lm
        })
        
    out_path = os.path.join(r"d:\LSC\webapp", "lsc_motion_dummy.json")
    with open(out_path, 'w') as f:
        json.dump(frames_json, f)
        
    print(f"Exportado correctamente a: {out_path} para ser consumido por Kalidokit (Three.js)")

if __name__ == "__main__":
    export_dummy_web_json()
