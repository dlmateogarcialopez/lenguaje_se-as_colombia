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
DYNAMIC_LANDMARKS_DIR = "D:/LSC/pipeline_output/dynamic_landmarks"
OUTPUT_DIR       = "D:/LSC/pipeline_output/synthesized"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DYNAMIC_LANDMARKS_DIR, exist_ok=True)

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

# Global cache for the gloss index
GLOSS_INDEX = {}

def load_gloss_index():
    """Load the pre-built mapping of signs to available files."""
    global GLOSS_INDEX
    if os.path.exists(GLOSS_INDEX_PATH):
        try:
            with open(GLOSS_INDEX_PATH, "r", encoding="utf-8") as f:
                GLOSS_INDEX = json.load(f)
            print(f"DEBUG: Loaded gloss index with {len(GLOSS_INDEX)} entries.")
        except Exception as e:
            logging.error(f"Error loading gloss index: {e}")
    else:
        logging.warning("Gloss index not found. Falling back to disk scanning.")

# Load on module import
load_gloss_index()

# Linguistic Correction: Force specific samples known to be more accurate
# Format: { "GLOSS": {"participant": "xxxx", "rep": y} }
PREFERRED_SAMPLES = {
    "GRACIAS": {"participant": "0004", "rep": 3}, # Starts higher up (better chin reach)
}

# --- Procedural Month Configuration ---
MONTH_CONFIG = {
    "MES_ENERO":      {"letter": "LETRA_E", "motion": "circle", "pos": [0.15, 0.45]}, # Cheek
    "MES_FEBRERO":    {"letter": "LETRA_F", "motion": "horiz",  "pos": [0.00, 0.35]}, # Eyes
    "MES_MARZO":      {"letter": "LETRA_M", "motion": "static", "pos": [0.00, 0.55]}, # Chin
    "MES_ABRIL":      {"letter": "LETRA_A", "motion": "circle", "pos": [0.00, 0.25]}, # Forehead
    "MES_MAYO":       {"letter": "LETRA_M", "motion": "down",   "pos": [0.00, 0.25]}, # Descend from forehead
    "MES_JUNIO":      {"letter": "LETRA_J", "motion": "curve",  "pos": [0.10, 0.45]}, # Near nose
    "MES_JULIO":      {"letter": "LETRA_J", "motion": "j_path", "pos": [0.20, 0.50]}, # Free space
    "MES_AGOSTO":     {"letter": "LETRA_A", "motion": "static", "pos": [0.25, 0.60]}, # Shoulder
    "MES_SEPTIEMBRE": {"letter": "LETRA_S", "motion": "static", "pos": [0.20, 0.35]}, # Temple
    "MES_OCTUBRE":    {"letter": "LETRA_O", "motion": "circle", "pos": [0.10, 0.35]}, # Eye
    "MES_NOVIEMBRE":  {"letter": "LETRA_N", "motion": "static", "pos": [0.05, 0.45]}, # Nose
    "MES_DICIEMBRE":  {"letter": "LETRA_D", "motion": "down",   "pos": [0.00, 0.55]}, # Beard motion
}

def load_dynamic_csv(label: str) -> np.ndarray:
    """Load alphabet/numbers landmarks using a fast, non-pandas method."""
    fname = f"{label}.csv"
    path = os.path.join(DYNAMIC_LANDMARKS_DIR, fname)
    if not os.path.exists(path):
        return None
        
    try:
        # Binary load to avoid text-mode hangs on some Windows environments
        with open(path, "rb") as f:
            content = f.read().decode('utf-8', errors='ignore')
        
        lines = content.splitlines()
        if not lines: return None
        
        data = []
        # Skip header (line 0)
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) > 1:
                # Skip first part (index) and map to float
                # We use a faster list comprehension
                try:
                    vals = [float(x) if x else 0.0 for x in parts[1:]]
                    # Accept both full (1629) and pose-only (99)
                    if len(vals) in [1629, 99]:
                        data.append(vals)
                except ValueError:
                    continue
        
        if not data:
            print(f"WARNING: No valid data rows found in {fname} (Columns found: {len(parts) if 'parts' in locals() else 'Unknown'})")
            return None
            
        return np.array(data)
    except Exception as e:
        print(f"ERROR loading dynamic CSV {label}: {e}")
        return None

def synthesize_month_procedural(gloss: str):
    """Generates landmarks for a month by applying motion to a base letter."""
    cfg = MONTH_CONFIG.get(gloss)
    if not cfg:
        return np.zeros((FRAMES_PER_SIGN, 33, 3)), np.zeros((FRAMES_PER_SIGN, 468, 3)), np.zeros((FRAMES_PER_SIGN, 21, 3)), np.zeros((FRAMES_PER_SIGN, 21, 3))
    
    # 1. Load base letter data (as numpy array directly)
    data = load_dynamic_csv(cfg["letter"])
    if data is None:
        return np.zeros((FRAMES_PER_SIGN, 33, 3)), np.zeros((FRAMES_PER_SIGN, 468, 3)), np.zeros((FRAMES_PER_SIGN, 21, 3)), np.zeros((FRAMES_PER_SIGN, 21, 3))
    
    # Custom Sampling (simulating csv_to_landmark_array logic but with flat data)
    frames = data.shape[0]
    indices = np.linspace(0, frames - 1, FRAMES_PER_SIGN, dtype=int)
    sampled = data[indices]
    
    # Reshape features into structured landmarks
    feat_count = sampled.shape[1]
    
    if feat_count == 1629:
        body  = sampled[:, 0:99].reshape((FRAMES_PER_SIGN, 33, 3))
        face  = sampled[:, 99:1503].reshape((FRAMES_PER_SIGN, 468, 3))
        rhand = sampled[:, 1503:1566].reshape((FRAMES_PER_SIGN, 21, 3))
        lhand = sampled[:, 1566:1629].reshape((FRAMES_PER_SIGN, 21, 3))
    else:
        # Pose only (99)
        body  = sampled[:, 0:99].reshape((FRAMES_PER_SIGN, 33, 3))
        face  = np.zeros((FRAMES_PER_SIGN, 468, 3))
        rhand = np.zeros((FRAMES_PER_SIGN, 21, 3))
        lhand = np.zeros((FRAMES_PER_SIGN, 21, 3))

    # 2. Apply target position (offset)
    # We move the right hand and the arm (wrist/elbow) to the target position
    target = cfg["pos"] # [x_offset, y_target]
    
    for t in range(FRAMES_PER_SIGN):
        # Procedural Offset Logic
        dx, dy = 0, 0
        progress = t / FRAMES_PER_SIGN
        
        if cfg["motion"] == "circle":
            dx = 0.05 * np.cos(progress * 2 * np.pi)
            dy = 0.05 * np.sin(progress * 2 * np.pi)
        elif cfg["motion"] == "horiz":
            dx = 0.10 * np.sin(progress * 2 * np.pi)
        elif cfg["motion"] == "down":
            dy = 0.20 * progress
        elif cfg["motion"] == "curve":
            dx = 0.05 * np.sin(progress * np.pi)
            dy = 0.10 * progress
        elif cfg["motion"] == "j_path":
            dx = 0.05 * np.cos(progress * np.pi)
            dy = 0.20 * progress if progress > 0.5 else 0
            
        # Target center (Head is approx 0.5, 0.4)
        base_x = 0.5 + target[0]
        base_y = target[1]
        
        # Move the right hand (rhand and pose 16)
        # Calculate current hand center in base landmarks to find translation
        current_wrist = rhand[t][0] # Landmark 0 is wrist
        offset_x = (base_x + dx) - current_wrist[0]
        offset_y = (base_y + dy) - current_wrist[1]
        
        rhand[t][:, 0] += offset_x
        rhand[t][:, 1] += offset_y
        
        # Move pose hand landmarks (16, 18, 20, 22)
        hand_indices = [16, 18, 20, 22]
        for idx in hand_indices:
            body[t][idx][0] += offset_x
            body[t][idx][1] += offset_y
            
        # Adjust Elbow (14) to follow roughly
        body[t][14][0] = body[t][16][0] + 0.1
        body[t][14][1] = body[t][16][1] + 0.2
        
    return body, face, rhand, lhand


def load_lsc50_csv(directory: str, sign_id: str, gloss: str = None) -> pd.DataFrame | None:
    """Load a specific sign sample from LSC50 CSV landmark files using the index."""
    
    # Select best file from index if available
    target_file = None
    
    if gloss and gloss.upper() in GLOSS_INDEX:
        info = GLOSS_INDEX[gloss.upper()]
        
        # Determine which subdir we are in
        key = "body"
        if "HANDS_LANDMARKS/LEFT" in directory: key = "lhand"
        elif "HANDS_LANDMARKS/RIGHT" in directory: key = "rhand"
        elif "FACE_LANDMARKS" in directory: key = "face"
        
        files = info.get(key, [])
        if files:
            # 1. Try to find the preferred one if specified
            if gloss.upper() in PREFERRED_SAMPLES:
                pref = PREFERRED_SAMPLES[gloss.upper()]
                match_name = f"{sign_id}_{pref['participant']}_{str(pref['rep']).zfill(4)}.csv"
                if match_name in files:
                    target_file = match_name
            
            # 2. Try default 0000_0000
            if not target_file:
                default_name = f"{sign_id}_0000_0000.csv"
                if default_name in files:
                    target_file = default_name
            
            # 3. Just take the first one
            if not target_file:
                target_file = files[0]
                
    if target_file:
        path = os.path.join(directory, target_file)
        if os.path.exists(path):
            return pd.read_csv(path).fillna(0)

    # 4. Emergency fallback (Legacy behavior - only if index is missing)
    if not GLOSS_INDEX:
        for p in range(5): # Limit scan depth for speed
            for r in range(2):
                fname = f"{sign_id}_{str(p).zfill(4)}_{str(r).zfill(4)}.csv"
                path = os.path.join(directory, fname)
                if os.path.exists(path):
                    return pd.read_csv(path).fillna(0)
    
    return None

def csv_to_landmark_array(df, num_landmarks: int, target_frames: int = FRAMES_PER_SIGN) -> np.ndarray:
    """
    Converts a CSV (Pandas or NumPy) to a normalized (T, N, 3) numpy array.
    Handles variable-length sequences by uniform sampling.
    """
    if df is None:
        return np.zeros((target_frames, num_landmarks, 3))
    
    # CASE 1: NumPy Array (Already loaded/processed)
    if isinstance(df, np.ndarray):
        if df.size == 0:
            return np.zeros((target_frames, num_landmarks, 3))
        
        frames = df.shape[0]
        indices = np.linspace(0, frames - 1, target_frames, dtype=int)
        sampled = df[indices]
        
        # If it's already 3D (T, N, 3), return it
        if len(sampled.shape) == 3:
            return sampled
            
        # If it's 2D (T, Features), we need to check if features match mapping
        # For LSC70/LSC50 typical shape is (T, 1629) or (T, N*3)
        if len(sampled.shape) == 2:
            feat_count = sampled.shape[1]
            if feat_count == 1629:
                # Standard LSC layout: Pose(99), Face(1404), RHand(63), LHand(63)
                if num_landmarks == 33:   return sampled[:, 0:99].reshape((target_frames, 33, 3))
                elif num_landmarks == 468: return sampled[:, 99:1503].reshape((target_frames, 468, 3))
                elif num_landmarks == 21:  
                    # This is ambiguous between R/L hand. 
                    # For safety, we'll try to guess based on context or just return zeros.
                    # But actually we often call it specifically.
                    return sampled[:, 1503:1566].reshape((target_frames, 21, 3)) 
            else:
                # Generic fallback if features = num_landmarks * 3
                n_available = feat_count // 3
                n_take = min(num_landmarks, n_available)
                res = np.zeros((target_frames, num_landmarks, 3))
                res[:, :n_take, :] = sampled[:, :n_take*3].reshape((target_frames, n_take, 3))
                return res
        
        return np.zeros((target_frames, num_landmarks, 3))

    # CASE 2: Pandas DataFrame
    if df.empty:
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
    print(f"INFO: Starting synthesis for {len(glosses)} glosses...")
    all_frames = []
    prev_body = None
    
    # Cache for loaded landmark arrays to speed up repetitive glosses (e.g. NUMERO_0)
    landmark_cache = {}
    
    for i, gloss in enumerate(glosses):
        gloss_upper = gloss.upper()
        
        # Check cache first
        if gloss_upper in landmark_cache:
            body_arr, face_arr, rhand_arr, lhand_arr = landmark_cache[gloss_upper]
        else:
            # 1. Procedural Month Synthesis
            if gloss_upper.startswith("MES_"):
                body_arr, face_arr, rhand_arr, lhand_arr = synthesize_month_procedural(gloss_upper)
            else:
                sign_id = GLOSS_TO_SIGNID.get(gloss_upper)
                dynamic_df = None
                # Priority: Check if it's a dynamic prefix (LETRA_ or NUMERO_)
                if gloss_upper.startswith("LETRA_") or gloss_upper.startswith("NUMERO_"):
                    dynamic_df = load_dynamic_csv(gloss_upper)
                    
                if dynamic_df is not None:
                    # Use dynamic landmarks
                    body_arr  = csv_to_landmark_array(dynamic_df, 33, FRAMES_PER_SIGN)
                    face_arr  = csv_to_landmark_array(dynamic_df, 468, FRAMES_PER_SIGN) 
                    rhand_arr = csv_to_landmark_array(dynamic_df, 21, FRAMES_PER_SIGN)
                    lhand_arr = csv_to_landmark_array(dynamic_df, 21, FRAMES_PER_SIGN)
                elif sign_id is not None:
                    # Load landmarks from LSC50
                    body_df  = load_lsc50_csv(LSC50_BODY_DIR,  sign_id, gloss)
                    face_df  = load_lsc50_csv(LSC50_FACE_DIR,  sign_id, gloss)
                    rhand_df = load_lsc50_csv(LSC50_RHAND_DIR, sign_id, gloss)
                    lhand_df = load_lsc50_csv(LSC50_LHAND_DIR, sign_id, gloss)
                    
                    if body_df is None:
                        logging.warning(f"No samples found for {gloss} (sign_id={sign_id}) - using placeholder")
                        body_arr  = np.zeros((FRAMES_PER_SIGN, 33, 3))
                        face_arr  = np.zeros((FRAMES_PER_SIGN, 468, 3))
                        rhand_arr = np.zeros((FRAMES_PER_SIGN, 21, 3))
                        lhand_arr = np.zeros((FRAMES_PER_SIGN, 21, 3))
                    else:
                        body_arr  = csv_to_landmark_array(body_df, 33, FRAMES_PER_SIGN)
                        face_arr  = csv_to_landmark_array(face_df, 468, FRAMES_PER_SIGN)
                        rhand_arr = csv_to_landmark_array(rhand_df, 21, FRAMES_PER_SIGN)
                        lhand_arr = csv_to_landmark_array(lhand_df, 21, FRAMES_PER_SIGN)
                else:
                    logging.warning(f"Gloss {gloss} not found in any dataset - using placeholder")
                    body_arr  = np.zeros((FRAMES_PER_SIGN, 33, 3))
                    face_arr  = np.zeros((FRAMES_PER_SIGN, 468, 3))
                    rhand_arr = np.zeros((FRAMES_PER_SIGN, 21, 3))
                    lhand_arr = np.zeros((FRAMES_PER_SIGN, 21, 3))
            
            # Store in cache
            landmark_cache[gloss_upper] = (body_arr, face_arr, rhand_arr, lhand_arr)
        
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
            # Optimization: Only send every 5th face landmark to reduce JSON size and serialization time
            # The frontend (drawFace) only uses every 5th one anyway.
            face_subset = face_arr[t][::5] 
            
            all_frames.append({
                "gloss": gloss,
                "poseLandmarks": array_to_landmark_list(body_arr[t]),
                "pose3DLandmarks": array_to_landmark_list(body_arr[t]),
                "leftHandLandmarks": array_to_landmark_list(lhand_arr[t]),
                "rightHandLandmarks": array_to_landmark_list(rhand_arr[t]),
                "faceLandmarks": array_to_landmark_list(face_subset),
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
