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

MONTH_CONFIG = {
    # Positions are relative to nose. -X is the avatar's Right side, +X is Left side, +Y is down.
    
    # Enero: 'E' at right cheek
    "MES_ENERO":      {"letter": "LETRA_E", "motion": "circle", "pos": [ -0.10, -0.05]},
    
    # Febrero: Index up ('F' or 'Numero_1') at right cheek/temple
    "MES_FEBRERO":    {"letter": "LETRA_F", "motion": "horiz",  "pos": [ -0.10, -0.15]},
    
    # Marzo: 'M' or 3-fingers at right chest/shoulder
    "MES_MARZO":      {"letter": "LETRA_M", "motion": "down", "pos": [ -0.12,  0.10]},
    
    # Abril: Two hands in 'C' shape slowly morphing into 'O' near the face while moving inwards
    "MES_ABRIL":      {"letter": "LETRA_C", "left_letter": "LETRA_C", "end_letter": "LETRA_O", "left_end_letter": "LETRA_O", "motion": "pinch", "pos": [ -0.15, -0.05], "lpos": [ 0.15, -0.05]},
    
    # Mayo: Two hands in fists (S) at cheeks!
    "MES_MAYO":       {"letter": "LETRA_S", "left_letter": "LETRA_S", "motion": "circle", "pos": [ -0.12, -0.05], "lpos": [0.12, -0.05]},
    
    # Junio: 'J' curving down at chest/shoulder
    "MES_JUNIO":      {"letter": "LETRA_J", "motion": "curve",  "pos": [ -0.12, 0.05]},
    
    # Julio: 'L' or expanded 'J' at right chest
    "MES_JULIO":      {"letter": "LETRA_L", "motion": "j_path", "pos": [ -0.15, 0.10]},
    
    # Agosto: 'A' at right shoulder
    "MES_AGOSTO":     {"letter": "LETRA_A", "motion": "static", "pos": [ -0.18,  0.12]},
    
    # Septiembre: Index '1' at nose
    "MES_SEPTIEMBRE": {"letter": "NUMERO_1", "motion": "down", "pos": [ 0.0, -0.02]},
    
    # Octubre: 'O' near right jaw/shoulder
    "MES_OCTUBRE":    {"letter": "LETRA_O", "motion": "circle", "pos": [ -0.15, -0.05]},
    
    # Noviembre: Index '1' at right neck
    "MES_NOVIEMBRE":  {"letter": "NUMERO_1", "motion": "static", "pos": [ -0.08, 0.08]},
    
    # Diciembre: Two hands flat 'B' rubbing at stomach 
    "MES_DICIEMBRE":  {"letter": "LETRA_B", "left_letter": "LETRA_B", "motion": "horiz", "pos": [ -0.03, 0.28], "lpos": [ 0.03, 0.30]},
    
    # === LSCPROPIO MISSING PROCEDURALS ===
    "CELULAR": {"letter": "LETRA_C", "motion": "static", "pos": [ -0.15, -0.05]},
    "INTERNET": {"letter": "LETRA_I", "left_letter": "LETRA_I", "motion": "horiz", "pos": [ -0.10, 0.10], "lpos": [ 0.10, 0.10]},
    "TELEVISION": {"letter": "LETRA_T", "left_letter": "LETRA_T", "motion": "horiz", "pos": [ -0.15, 0.08], "lpos": [ 0.15, 0.08]},
    "COMPUTADOR": {"letter": "LETRA_C", "left_letter": "LETRA_C", "motion": "horiz", "pos": [ -0.12, 0.15], "lpos": [ 0.12, 0.15]},
    "VISITAR": {"letter": "LETRA_V", "motion": "j_path", "pos": [ -0.05, 0.05]},
    "A VECES": {"letter": "LETRA_V", "motion": "circle", "pos": [ -0.05, 0.10]},
    "ABAJO": {"letter": "LETRA_A", "motion": "down", "pos": [ -0.05, 0.15]},
    "AÑO": {"letter": "LETRA_A", "left_letter": "LETRA_A", "motion": "circle", "pos": [ -0.05, 0.15], "lpos": [ 0.05, 0.15]},
    "CAMARA FOTOGRAFICA": {"letter": "LETRA_C", "left_letter": "LETRA_C", "motion": "static", "pos": [ -0.10, -0.05], "lpos": [ 0.10, -0.05]},
    "DOMINGO": {"letter": "LETRA_D", "motion": "circle", "pos": [ -0.10, 0.10]},
    "ENCONTRAR": {"letter": "LETRA_E", "motion": "j_path", "pos": [ 0.0, 0.10]},
    "ENERGIA ELECTRICA": {"letter": "LETRA_E", "motion": "horiz", "pos": [ 0.0, 0.05]},
    "ESCOGER": {"letter": "LETRA_E", "motion": "down", "pos": [ -0.05, 0.05]},
    "FECHA": {"letter": "LETRA_F", "motion": "static", "pos": [ -0.05, 0.10]},
    "IMPRESORA": {"letter": "LETRA_I", "motion": "horiz", "pos": [ 0.0, 0.15]},
    "INTERPRETAR": {"letter": "LETRA_I", "left_letter": "LETRA_I", "motion": "circle", "pos": [ -0.10, 0.15], "lpos": [ 0.10, 0.15]},
    "JUEVES": {"letter": "LETRA_J", "motion": "j_path", "pos": [ -0.10, 0.05]},
    "LUNES": {"letter": "LETRA_L", "motion": "circle", "pos": [ -0.10, 0.05]},
    "MARTES": {"letter": "LETRA_M", "motion": "circle", "pos": [ -0.10, 0.05]},
    "MIERCOLES": {"letter": "LETRA_M", "motion": "horiz", "pos": [ -0.10, 0.05]},
    "SABADO": {"letter": "LETRA_S", "motion": "circle", "pos": [ -0.10, 0.05]},
    "VIERNES": {"letter": "LETRA_V", "motion": "circle", "pos": [ -0.10, 0.05]},
    "PAGAR": {"letter": "LETRA_P", "motion": "down", "pos": [ 0.0, 0.10]},
    "PARECER": {"letter": "LETRA_P", "motion": "j_path", "pos": [ -0.10, 0.05]},
    "PARTICIPAR": {"letter": "LETRA_P", "left_letter": "LETRA_P", "motion": "horiz", "pos": [ -0.10, 0.15], "lpos": [ 0.10, 0.15]},
    "PEDIR": {"letter": "LETRA_P", "motion": "circle", "pos": [ -0.05, 0.15]},
    "PODER": {"letter": "LETRA_P", "left_letter": "LETRA_P", "motion": "down", "pos": [ -0.10, 0.15], "lpos": [ 0.10, 0.15]},
    "PRECIO": {"letter": "LETRA_P", "motion": "static", "pos": [ -0.05, 0.10]},
    "TELEFONO": {"letter": "LETRA_Y", "motion": "static", "pos": [ -0.15, -0.05]},
    "TIEMPO": {"letter": "LETRA_T", "motion": "circle", "pos": [ -0.10, 0.15]},
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
        # The base file usually stores the dominant hand shape in the right hand column
        lhand = sampled[:, 99:162].reshape((FRAMES_PER_SIGN, 21, 3))
        rhand = sampled[:, 162:225].reshape((FRAMES_PER_SIGN, 21, 3))
        face  = sampled[:, 225:1629].reshape((FRAMES_PER_SIGN, 468, 3))
    else:
        # Pose only (99)
        body  = sampled[:, 0:99].reshape((FRAMES_PER_SIGN, 33, 3))
        lhand = np.zeros((FRAMES_PER_SIGN, 21, 3))
        rhand = np.zeros((FRAMES_PER_SIGN, 21, 3))
        face  = np.zeros((FRAMES_PER_SIGN, 468, 3))

    # 1.5 Load Left Hand base if two-handed sign
    left_letter = cfg.get("left_letter")
    if left_letter:
        left_data = load_dynamic_csv(left_letter)
        if left_data is not None:
            left_frames = left_data.shape[0]
            left_indices = np.linspace(0, left_frames - 1, FRAMES_PER_SIGN, dtype=int)
            left_sampled = left_data[left_indices]
            if left_sampled.shape[1] == 1629:
                # Often the shape we want for left hand was recorded with the right hand 
                # (e.g. LETRA_B is usually signed right-handed). So we extract the Right Hand column
                # and put it into the Left Hand array! Then we just mirror X coords!
                base_rh = left_sampled[:, 162:225].reshape((FRAMES_PER_SIGN, 21, 3))
                # Mirror X around the origin to make a right hand look like a left hand
                base_rh[:, :, 0] = -base_rh[:, :, 0]
                lhand = base_rh
                
    # 1.6 Load End Shapes for Morphing
    end_rhand_base = None
    if cfg.get("end_letter"):
        e_data = load_dynamic_csv(cfg["end_letter"])
        if e_data is not None:
            e_frames = e_data.shape[0]
            e_sampled = e_data[np.linspace(0, e_frames - 1, FRAMES_PER_SIGN, dtype=int)]
            if e_sampled.shape[1] == 1629:
                end_rhand_base = e_sampled[:, 162:225].reshape((FRAMES_PER_SIGN, 21, 3))
                
    end_lhand_base = None
    if cfg.get("left_end_letter"):
        el_data = load_dynamic_csv(cfg["left_end_letter"])
        if el_data is not None:
            el_frames = el_data.shape[0]
            el_sampled = el_data[np.linspace(0, el_frames - 1, FRAMES_PER_SIGN, dtype=int)]
            if el_sampled.shape[1] == 1629:
                el_rh = el_sampled[:, 162:225].reshape((FRAMES_PER_SIGN, 21, 3))
                el_rh[:, :, 0] = -el_rh[:, :, 0]
                end_lhand_base = el_rh

    # 2. Apply target position RELATIVE to the avatar's nose (pose landmark 0)
    # This ensures the hand stays anchored to the body regardless of the base landmark values.
    # target = [x_offset_from_nose, y_offset_from_nose]
    # Positive x = right side of face, positive y = below nose
    target = cfg["pos"]  # [x_offset, y_offset] relative to nose

    for t in range(FRAMES_PER_SIGN):
        # -- Motion modifier (small oscillation on top of the base position) --
        dx, dy, l_dx, l_dy = 0.0, 0.0, 0.0, 0.0
        progress = t / FRAMES_PER_SIGN

        if cfg["motion"] == "circle":
            dx = 0.04 * np.cos(progress * 2 * np.pi)
            dy = 0.04 * np.sin(progress * 2 * np.pi)
        elif cfg["motion"] == "horiz":
            dx = 0.06 * np.sin(progress * 2 * np.pi)
        elif cfg["motion"] == "down":
            dy = 0.10 * progress
        elif cfg["motion"] == "curve":
            dx = 0.04 * np.sin(progress * np.pi)
            dy = 0.06 * progress
        elif cfg["motion"] == "j_path":
            dx = 0.04 * np.cos(progress * np.pi)
            dy = 0.10 * progress if progress > 0.5 else 0
        elif cfg["motion"] == "pinch":
            dx = 0.06 * progress      # Right hand moves left (inwards)
            dy = 0.0
            l_dx = -0.06 * progress   # Left hand moves right (inwards)
            l_dy = 0.0
        else:
            dx, dy, l_dx, l_dy = 0.0, 0.0, 0.0, 0.0

        # -- Anchor: use the nose (pose landmark 0) as the reference centre --
        # In MediaPipe normalised coords, the nose sits roughly at (0.5, 0.35)
        # but we use the actual value from the loaded data so it adapts properly.
        nose_x = body[t][0][0]  # pose landmark 0 = nose
        nose_y = body[t][0][1]

        # Desired wrist world position = nose + face-region offset + motion
        desired_x = nose_x + target[0] + dx
        desired_y = nose_y + target[1] + dy

        # Morphological interpolation (shape shifting)
        if end_rhand_base is not None:
            # Align target shape's wrist definitively to the start shape's wrist
            aligned_end_rhand = end_rhand_base[t] - end_rhand_base[t][0] + rhand[t][0]
            # Shift between initial hand shape and target hand shape smoothly
            rhand[t] = rhand[t] * (1.0 - progress) + aligned_end_rhand * progress

        # -- Translate right hand so its wrist reaches the desired position --
        # ALWAYS use Pose Wrist (16) as the reliable physical anchor, since 
        # rhand[t][0] might be [0,0,0] if MediaPipe lost high-res hand tracking.
        current_wrist = body[t][16]
        offset_x = desired_x - current_wrist[0]
        offset_y = desired_y - current_wrist[1]

        # Inject artificial Z-depth reach as hands approach center
        # Max reach is at 50% progress, pushing arms FORWARD toward camera (negative Z)
        offset_z = -0.22 * np.sin(progress * np.pi)

        # Shift the detailed hand mesh
        # If the hand mesh was tracked, it shifts perfectly. 
        # If it was [0,0,0], it collapses into a point at the wrist.
        rhand[t][:, 0] += offset_x
        rhand[t][:, 1] += offset_y
        rhand[t][:, 2] += offset_z

        # Adjust arm to follow wrist
        # Pose right wrist is index 16
        for idx in [16, 18, 20, 22]:
            body[t][idx][0] += offset_x
            body[t][idx][1] += offset_y
            body[t][idx][2] += offset_z

        # Adjust elbow (14) to trail behind wrist naturally with Z push
        body[t][14][0] = body[t][16][0] + 0.08
        body[t][14][1] = body[t][16][1] + 0.15
        body[t][14][2] = body[t][16][2] + 0.05

        # -- Left Hand processing --
        if left_letter:
            # left target offset from nose
            ltx = cfg.get("lpos", [0])[0]
            lty = cfg.get("lpos", [0, 0])[1] if len(cfg.get("lpos", [0])) > 1 else 0
            
            l_desired_x = nose_x + ltx + l_dx
            l_desired_y = nose_y + lty + l_dy
            
            l_current_wrist = body[t][15] # Left pose wrist
            
            # Morphological interpolation (shape shifting)
            if end_lhand_base is not None:
                aligned_end_lhand = end_lhand_base[t] - end_lhand_base[t][0] + lhand[t][0]
                lhand[t] = lhand[t] * (1.0 - progress) + aligned_end_lhand * progress
                
            l_offset_x = l_desired_x - l_current_wrist[0]
            l_offset_y = l_desired_y - l_current_wrist[1]
            
            lhand[t][:, 0] += l_offset_x
            lhand[t][:, 1] += l_offset_y
            lhand[t][:, 2] += offset_z
            
            for idx in [15, 17, 19, 21]:
                body[t][idx][0] += l_offset_x
                body[t][idx][1] += l_offset_y
                body[t][idx][2] += offset_z
                
            # Adjust left elbow (13)
            body[t][13][0] = body[t][15][0] - 0.08
            body[t][13][1] = body[t][15][1] + 0.15
            body[t][13][2] = body[t][15][2] + 0.05
        else:
            # Single-handed sign: Force the left arm to rest down by the waist
            # and ignore any spurious finger movements from the base file.
            lhand[t] = np.zeros((21, 3))
            
            # Left hip is 23
            hip_x, hip_y = body[t][23][0], body[t][23][1]
            rest_x = hip_x + 0.10
            rest_y = hip_y + 0.05
            
            l_current_wrist = body[t][15]
            l_offset_x = rest_x - l_current_wrist[0]
            l_offset_y = rest_y - l_current_wrist[1]
            
            for idx in [15, 17, 19, 21]:
                body[t][idx][0] += l_offset_x
                body[t][idx][1] += l_offset_y
                
            # Left elbow (13)
            body[t][13][0] = body[t][15][0] + 0.05
            body[t][13][1] = body[t][15][1] - 0.10 # elbow slightly above wrist when resting
            
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
            sign_id = GLOSS_TO_SIGNID.get(gloss_upper)
            dynamic_df = None
            
            # 1. Priority: Check if any dynamic/override file exists natively
            if os.path.exists(os.path.join(DYNAMIC_LANDMARKS_DIR, f"{gloss_upper}.csv")):
                dynamic_df = load_dynamic_csv(gloss_upper)
                
            # 2. Check fallback for Procedural synthesis
            if dynamic_df is None and (gloss_upper.startswith("MES_") or gloss_upper in MONTH_CONFIG):
                body_arr, face_arr, rhand_arr, lhand_arr = synthesize_month_procedural(gloss_upper)
            elif dynamic_df is not None:
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
