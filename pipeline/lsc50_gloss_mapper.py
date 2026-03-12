"""
lsc50_gloss_mapper.py

Reads the LSC50 dataset metadata to map numeric sign IDs → Spanish labels.
Then builds a gloss index.

LSC50 naming convention: PARTICIPANT_SIGNID_REPETITION.csv
  - PARTICIPANT: 0000–0019 (20 participants)
  - SIGNID: 0000–0049 (50 signs)
  - REPETITION: 0000–0003 (4 reps per participant per sign)

So 20 × 50 × 4 = 4000 files total, but only 1000 in BODY_LANDMARKS
(because 10 participants or fewer have body data)
"""
import os
import json
import re
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ── Paths on D: drive ────────────────────────────────────────────────────────
LSC50_BASE      = "D:/LSC/LSC50"
LSC50_BODY_DIR  = f"{LSC50_BASE}/LANDMARKS/BODY_LANDMARKS"
LSC50_FACE_DIR  = f"{LSC50_BASE}/LANDMARKS/FACE_LANDMARKS"
LSC50_RHAND_DIR = f"{LSC50_BASE}/LANDMARKS/HANDS_LANDMARKS/RIGHT_HAND_LANDMARKS"
LSC50_LHAND_DIR = f"{LSC50_BASE}/LANDMARKS/HANDS_LANDMARKS/LEFT_HAND_LANDMARKS"
OUTPUT_DIR      = "D:/LSC/pipeline_output/gloss_index"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Full 50-sign vocabulary for LSC50 ────────────────────────────────────────
LSC50_SIGN_LABELS = {
    "0000": "AMIGO",    "0001": "AMOR",     "0002": "AYUDA",
    "0003": "BIEN",     "0004": "BUENAS",   "0005": "BUENAS_NOCHES",
    "0006": "CAFE",     "0007": "CAMINAR",  "0008": "CASA",
    "0009": "COMER",    "0010": "COMUNICACION", "0011": "CORRER",
    "0012": "DINERO",   "0013": "DORMIR",   "0014": "ENFERMO",
    "0015": "ESTUDIANTE","0016": "FAMILIA", "0017": "FELIZ",
    "0018": "GRACIAS",  "0019": "GRANDE",   "0020": "HABLAR",
    "0021": "HOLA",     "0022": "HOMBRE",   "0023": "HOSPITAL",
    "0024": "JUGAR",    "0025": "LECHE",    "0026": "LLAMAR",
    "0027": "LUNES",    "0028": "MADRE",    "0029": "MAL",
    "0030": "MAÑANA",   "0031": "MES",      "0032": "MIERCOLES",
    "0033": "MUJER",    "0034": "NECESITAR","0035": "NIÑO",
    "0036": "NOCHE",    "0037": "NUMERO",   "0038": "PADRE",
    "0039": "PAN",      "0040": "PEQUEÑO",  "0041": "PERDON",
    "0042": "POLICIA",  "0043": "PREGUNTA", "0044": "SABADO",
    "0045": "SALUD",    "0046": "TARDE",    "0047": "TRABAJO",
    "0048": "VIERNES",  "0049": "VIVIR",
}

PATTERN = re.compile(r'^(\d{4})_(\d{4})_(\d{4})\.csv$')
# Group 1: sign_id, Group 2: participant, Group 3: rep

def scan_directory(directory: str) -> dict:
    """Scans a landmark directory and returns { label: [filename...] }"""
    if not os.path.exists(directory):
        logging.warning(f"Dir not found: {directory}")
        return {}
    
    files = sorted(os.listdir(directory))
    logging.info(f"  Scanning {directory}: {len(files)} files")
    
    index = defaultdict(list)
    for fname in files:
        m = PATTERN.match(fname)
        if not m:
            continue
        sign_id = m.group(1)
        label = LSC50_SIGN_LABELS.get(sign_id, f"SIGN_{sign_id}")
        index[label].append(fname)
    return dict(index)


def build_full_gloss_index() -> dict:
    """Builds the combined gloss index from all available landmark directories."""
    logging.info("Building full LSC50 gloss index...")
    
    body_index  = scan_directory(LSC50_BODY_DIR)
    face_index  = scan_directory(LSC50_FACE_DIR)
    rhand_index = scan_directory(LSC50_RHAND_DIR)
    lhand_index = scan_directory(LSC50_LHAND_DIR)
    
    # Merge: a sign is "available" if it has body or hand data
    all_labels = set(body_index) | set(face_index) | set(rhand_index) | set(lhand_index)
    logging.info(f"\nTotal available signs: {len(all_labels)}")
    
    combined = {}
    for label in sorted(all_labels):
        combined[label] = {
            "body":   body_index.get(label, []),
            "face":   face_index.get(label, []),
            "rhand":  rhand_index.get(label, []),
            "lhand":  lhand_index.get(label, []),
        }
        count = len(combined[label]["body"]) or len(combined[label]["rhand"])
        logging.info(f"  {label}: {count} samples")
    
    return combined


def save_indexes(combined: dict):
    # Save full combined index
    full_path = f"{OUTPUT_DIR}/gloss_index_full.json"
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    logging.info(f"\nFull index saved: {full_path}")
    
    # Save simple vocabulary list
    vocab = sorted(combined.keys())
    vocab_path = f"{OUTPUT_DIR}/vocabulary.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)
    logging.info(f"Vocabulary ({len(vocab)} signs) saved: {vocab_path}")


if __name__ == "__main__":
    combined = build_full_gloss_index()
    if combined:
        save_indexes(combined)
