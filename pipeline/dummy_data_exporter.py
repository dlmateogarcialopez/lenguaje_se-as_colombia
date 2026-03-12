import os
import json
import numpy as np
import config
import pickle

def export_dummy_web_json():
    print("Extrayendo secuencia de la Matriz NumPy para el reproductor WebGL...")
    
    tensor_path = os.path.join(config.OUTPUT_DIR, "LSC_dataset_unificado.npy")
    if not os.path.exists(tensor_path):
        print("Error: No se encontró el dataset unificado.npy")
        return
        
    X = np.load(tensor_path)
    label_path = os.path.join(config.OUTPUT_DIR, "LSC_labels.npy")
    le_path = os.path.join(config.OUTPUT_DIR, "label_encoder.pkl")
    
    Y = np.load(label_path) if os.path.exists(label_path) else None
    le = None
    if os.path.exists(le_path):
        with open(le_path, 'rb') as f:
            le = pickle.load(f)

    # --- EXPERT SELECTION: High Motion in Hands + Valid Face ---
    best_idx = 0
    max_score = 0
    final_label = "Desconocida"
    
    print("Iniciando búsqueda experta de muestras (Filtrado por Varianza)...")
    
    for i in range(min(200, len(X))):
        sample = X[i]
        
        # Slices: Pose(0:33*3), LHand(99:162), RHand(162:225), Face(225:1629)
        l_hand = sample[:, 33*3 : (33+21)*3]
        r_hand = sample[:, (33+21)*3 : (33+21+21)*3]
        face = sample[:, (33+21+21)*3:]
        
        # Calculate articulation variance
        h_score = np.std(l_hand) + np.std(r_hand) 
        f_score = np.std(face)
        
        # Penalty for samples with near-zero hand motion (likely filler/background)
        if h_score < 0.05: continue 
        
        total_score = (h_score * 5) + (f_score * 2)
        
        if total_score > max_score:
            # Check label
            lbl = "Desconocida"
            if Y is not None and le is not None:
                idx = np.argmax(Y[i])
                lbl = le.inverse_transform([idx])[0]
            
            # Avoid generic or padding labels if possible
            if lbl.lower() in ["padding", "0", "unknown"]: continue
            
            max_score = total_score
            best_idx = i
            final_label = lbl
            
    print(f"Muestra Profesional Seleccionada: Index {best_idx}")
    print(f"Etiqueta: {final_label} | Score de Movimiento: {max_score:.4f}")
    sample_seq = X[best_idx]
    label_name = final_label
    
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
    final_output = {
        "label": label_name,
        "index": int(best_idx),
        "frames": frames_json
    }
    with open(out_path, 'w') as f:
        json.dump(final_output, f)
        
    print(f"Exportado correctamente a: {out_path} para ser consumido por Kalidokit (Three.js)")

if __name__ == "__main__":
    export_dummy_web_json()
