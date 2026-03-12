"""
motion_synthesizer.py
Gloss Sequence → Continuous Landmark JSON for Web Animation

For each gloss in the sequence:
1. Load best sample from LSC50 CSV files
2. Normalize the coordinates
3. Blend transition frames between signs
4. Export as a unified JSON for the WebApp
"""
import os
import json
import logging
import numpy as np
import pandas as pd
from typing import List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Paths (D: drive) ---
LSC50_BODY_DIR   = "D:/LSC/LSC50/LANDMARKS/BODY_LANDMARKS"
LSC50_FACE_DIR   = "D:/LSC/LSC50/LANDMARKS/FACE_LANDMARKS"
LSC50_LHAND_DIR  = "D:/LSC/LSC50/LANDMARKS/HANDS_LANDMARKS/LEFT_HAND_LANDMARKS"
LSC50_RHAND_DIR  = "D:/LSC/LSC50/LANDMARKS/HANDS_LANDMARKS/RIGHT_HAND_LANDMARKS"
GLOSS_INDEX_PATH = "D:/LSC/pipeline_output/gloss_index/gloss_index_full.json"
OUTPUT_DIR       = "D:/LSC/pipeline_output/synthesized"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Sign ID mapping (matching lsc50_gloss_mapper.py)
GLOSS_TO_SIGNID = {
    "AMIGO": "0000", "AMOR": "0001", "AYUDA": "0002",
    "BIEN": "0003",  "BUENAS": "0004",
    "BUENAS_NOCHES": "0005", "CAFE": "0006", "CAMINAR": "0007",
    "CASA": "0008",  "COMER": "0009", "COMUNICACION": "0010",
    "CORRER": "0011", "DINERO": "0012", "DORMIR": "0013",
    "ENFERMO": "0014", "ESTUDIANTE": "0015", "FAMILIA": "0016",
    "FELIZ": "0017", "GRACIAS": "0018", "GRANDE": "0019",
    "HABLAR": "0020", "HOLA": "0021", "HOMBRE": "0022",
    "HOSPITAL": "0023", "JUGAR": "0024", "LECHE": "0025",
    "LLAMAR": "0026", "LUNES": "0027", "MADRE": "0028",
    "MAL": "0029", "MAÑANA": "0030", "MES": "0031",
    "MIERCOLES": "0032", "MUJER": "0033", "NECESITAR": "0034",
    "NIÑO": "0035", "NOCHE": "0036", "NUMERO": "0037",
    "PADRE": "0038", "PAN": "0039", "PEQUEÑO": "0040",
    "PERDON": "0041", "POLICIA": "0042", "PREGUNTA": "0043",
    "SABADO": "0044", "SALUD": "0045", "TARDE": "0046",
    "TRABAJO": "0047", "VIERNES": "0048", "VIVIR": "0049",
}

FRAMES_PER_SIGN    = 30   # Uniform sampling per sign
TRANSITION_FRAMES  = 5    # Blending frames between signs

def load_lsc50_csv(directory: str, sign_id: str, participant: str = "0000", rep: int = 0) -> pd.DataFrame | None:
    """Load a specific sign sample from LSC50 CSV landmark files."""
    # Pattern: {sign_id}_{participant}_{repetition}.csv
    fname = f"{sign_id}_{participant}_{str(rep).zfill(4)}.csv"
    path = os.path.join(directory, fname)
    if os.path.exists(path):
        return pd.read_csv(path).fillna(0)
    # Try any available participant
    for p in range(20):
        for r in range(4):
            fname = f"{sign_id}_{str(p).zfill(4)}_{str(r).zfill(4)}.csv"
            path = os.path.join(directory, fname)
            if os.path.exists(path):
                return pd.read_csv(path).fillna(0)
    return None

def csv_to_landmark_array(df: pd.DataFrame, num_landmarks: int, target_frames: int = FRAMES_PER_SIGN) -> np.ndarray:
    """
    Converts a CSV dataframe to a normalized (T, N, 3) numpy array.
    Handles variable-length sequences by uniform sampling.
    """
    if df is None or df.empty:
        return np.zeros((target_frames, num_landmarks, 3))
    
    # Try to find coordinate columns
    x_cols = [c for c in df.columns if c.lower().endswith('_x') or 'x_' in c.lower()]
    y_cols = [c for c in df.columns if c.lower().endswith('_y') or 'y_' in c.lower()]
    z_cols = [c for c in df.columns if c.lower().endswith('_z') or 'z_' in c.lower()]
    
    # If no explicit columns, try to parse from raw values
    if not x_cols:
        # Some CSVs have landmark_0_x format or just numbered cols
        cols = df.columns.tolist()
        n = len(cols) // 3
        x_cols = cols[0:n]
        y_cols = cols[n:2*n]
        z_cols = cols[2*n:3*n]
    
    # Limit to available landmarks
    n_lm = min(num_landmarks, len(x_cols))
    frames = len(df)
    
    if frames == 0:
        return np.zeros((target_frames, num_landmarks, 3))
    
    # Extract and stack XYZ
    try:
        X = df[x_cols[:n_lm]].values.astype(float)
        Y = df[y_cols[:n_lm]].values.astype(float)
        Z = df[z_cols[:n_lm]].values.astype(float)
        raw = np.stack([X, Y, Z], axis=-1)  # (frames, n_lm, 3)
    except Exception:
        return np.zeros((target_frames, num_landmarks, 3))
    
    # Uniform sampling to target_frames
    indices = np.linspace(0, frames - 1, target_frames, dtype=int)
    sampled = raw[indices]
    
    # Pad to full num_landmarks if less
    if n_lm < num_landmarks:
        pad = np.zeros((target_frames, num_landmarks - n_lm, 3))
        sampled = np.concatenate([sampled, pad], axis=1)
    
    return sampled

def linear_blend(start: np.ndarray, end: np.ndarray, n_frames: int) -> np.ndarray:
    """Create n_frames transition frames blending from end of sign A to start of sign B."""
    blended = []
    for t in range(n_frames):
        alpha = t / n_frames
        frame = (1 - alpha) * start + alpha * end
        blended.append(frame)
    return np.array(blended)

def array_to_landmark_list(arr: np.ndarray) -> list:
    """Convert (N, 3) numpy array to list of {x, y, z} dicts for JSON export."""
    return [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for p in arr]

def synthesize_sequence(glosses: List[str]) -> dict:
    """
    Main synthesis function.
    Takes a list of glosses and returns a JSON-serializable dict
    with sequential frames ready for WebGL rendering.
    """
    logging.info(f"Synthesizing {len(glosses)} glosses: {glosses}")
    
    all_frames = []
    prev_body = None
    
    for i, gloss in enumerate(glosses):
        sign_id = GLOSS_TO_SIGNID.get(gloss.upper())
        
        if sign_id is None:
            logging.warning(f"No sign ID for gloss: {gloss}")
            continue
        
        # Load body landmarks (33 pose, 21 Lhand, 21 Rhand)
        body_df = load_lsc50_csv(LSC50_BODY_DIR, sign_id)
        face_df = load_lsc50_csv(LSC50_FACE_DIR, sign_id)
        rhand_df = load_lsc50_csv(LSC50_RHAND_DIR, sign_id)
        lhand_df = load_lsc50_csv(LSC50_LHAND_DIR, sign_id)
        
        if body_df is None:
            logging.warning(f"No body data for {gloss} (sign_id={sign_id})")
            continue
        
        body_arr  = csv_to_landmark_array(body_df, 33, FRAMES_PER_SIGN)
        face_arr  = csv_to_landmark_array(face_df, 468, FRAMES_PER_SIGN)
        rhand_arr = csv_to_landmark_array(rhand_df, 21, FRAMES_PER_SIGN)
        lhand_arr = csv_to_landmark_array(lhand_df, 21, FRAMES_PER_SIGN)
        
        logging.info(f"  [{gloss}] body:{body_arr.shape}, face:{face_arr.shape}, rh:{rhand_arr.shape}, lh:{lhand_arr.shape}")
        
        # Add transition frames
        if prev_body is not None and TRANSITION_FRAMES > 0:
            trans = linear_blend(prev_body[-1], body_arr[0], TRANSITION_FRAMES)
            for t in range(TRANSITION_FRAMES):
                all_frames.append({
                    "gloss": f"{glosses[i-1]}→{gloss}",
                    "poseLandmarks": array_to_landmark_list(trans[t]),
                    "pose3DLandmarks": array_to_landmark_list(trans[t]),
                    "leftHandLandmarks": array_to_landmark_list(np.zeros((21, 3))),
                    "rightHandLandmarks": array_to_landmark_list(np.zeros((21, 3))),
                    "faceLandmarks": array_to_landmark_list(np.zeros((468, 3))),
                })
        
        # Add sign frames
        for t in range(FRAMES_PER_SIGN):
            all_frames.append({
                "gloss": gloss,
                "poseLandmarks": array_to_landmark_list(body_arr[t]),
                "pose3DLandmarks": array_to_landmark_list(body_arr[t]),
                "leftHandLandmarks": array_to_landmark_list(lhand_arr[t]),
                "rightHandLandmarks": array_to_landmark_list(rhand_arr[t]),
                "faceLandmarks": array_to_landmark_list(face_arr[t]),
            })
        
        prev_body = body_arr
        logging.info(f"  [{gloss}] → {FRAMES_PER_SIGN} frames added. Total: {len(all_frames)}")
    
    return {
        "glosses": glosses,
        "total_frames": len(all_frames),
        "frames_per_sign": FRAMES_PER_SIGN,
        "transition_frames": TRANSITION_FRAMES,
        "frames": all_frames,
    }

def export_to_json(result: dict, output_filename: str = "lsc_translation.json") -> str:
    """Export the synthesized sequence to JSON on D: drive."""
    out_path = os.path.join(OUTPUT_DIR, output_filename)
    # Also copy to webapp folder for immediate browser serving
    webapp_path = "D:/LSC/webapp/lsc_motion_dummy.json"
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    with open(webapp_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    
    logging.info(f"Exported: {out_path}")
    logging.info(f"WebApp copy: {webapp_path}")
    return out_path

if __name__ == "__main__":
    # Quick test
    test_glosses = ["HOLA", "GRACIAS", "AMIGO"]
    result = synthesize_sequence(test_glosses)
    logging.info(f"Total frames synthesized: {result['total_frames']}")
    out = export_to_json(result, "test_synthesis.json")
    print(f"\nSynthesis complete! Output: {out}")
    print(f"First frame gloss: {result['frames'][0]['gloss'] if result['frames'] else 'NONE'}")
