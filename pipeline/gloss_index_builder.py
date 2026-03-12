"""
gloss_index_builder.py
Builds a JSON gloss index from the LSC50 landmark dataset.
Scans all LANDMARK JSON files, extracts sign labels from filenames,
and builds a lookup dict: { "GLOSA_EN_MAYUSCULAS": [list of file paths] }
Stores everything in D: drive.
"""
import os
import json
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CONFIGURATION ---
LSC50_LANDMARKS_DIR = "D:/LSC/LSC50/LANDMARKS"
OUTPUT_DIR          = "D:/LSC/pipeline_output/gloss_index"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_label_from_filename(filename: str) -> str | None:
    """
    LSC50 filenames follow the pattern:
        <participant>_<SIGN_LABEL>_<repetition>.json
    or  <SIGN_LABEL>_<something>.json
    We try to extract the sign label from the filename stem.
    """
    stem = os.path.splitext(filename)[0]  # remove .json
    parts = stem.split("_")
    # Best heuristic: look for the biggest alphabetic chunk
    # Ignore purely numeric parts
    label_parts = []
    for p in parts:
        if not p.isdigit():
            label_parts.append(p.upper())
    # Most likely the middle segment IS the gloss
    if len(label_parts) >= 2:
        return "_".join(label_parts[1:-1]) if len(label_parts) > 2 else label_parts[0]
    elif len(label_parts) == 1:
        return label_parts[0]
    return None

def peek_landmark_structure(filepath: str) -> dict:
    """
    Reads just the first record of a landmark file to understand its schema.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list) and len(data) > 0:
        return {"type": "list", "len": len(data), "keys": list(data[0].keys()) if isinstance(data[0], dict) else []}
    elif isinstance(data, dict):
        return {"type": "dict", "keys": list(data.keys())}
    return {}

def build_gloss_index():
    """
    Main function: scans all LSC50 landmark files and builds the gloss index.
    """
    logging.info(f"Scanning: {LSC50_LANDMARKS_DIR}")
    
    if not os.path.exists(LSC50_LANDMARKS_DIR):
        logging.error(f"Directory not found: {LSC50_LANDMARKS_DIR}")
        return
    
    files = sorted([f for f in os.listdir(LSC50_LANDMARKS_DIR) if f.endswith(".json")])
    logging.info(f"Total landmark files found: {len(files)}")
    
    # Print first 10 filenames to understand naming convention
    logging.info("Sample filenames:")
    for f in files[:10]:
        logging.info(f"  {f}")
    
    # Build index: label -> list of file paths
    gloss_index = defaultdict(list)
    unknown_count = 0
    
    for fname in files:
        label = extract_label_from_filename(fname)
        if label:
            gloss_index[label].append(fname)
        else:
            unknown_count += 1
    
    logging.info(f"\nDistinct sign labels found: {len(gloss_index)}")
    logging.info(f"Unknown labels: {unknown_count}")
    logging.info(f"\nAll available signs:")
    for k in sorted(gloss_index.keys()):
        logging.info(f"  {k}: {len(gloss_index[k])} samples")
    
    # Save standard index
    index_path = os.path.join(OUTPUT_DIR, "gloss_index_lsc50.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(dict(gloss_index), f, indent=2, ensure_ascii=False)
    logging.info(f"\nGloss index saved: {index_path}")
    
    # Peek at structure of one file
    if files:
        sample_path = os.path.join(LSC50_LANDMARKS_DIR, files[0])
        structure = peek_landmark_structure(sample_path)
        logging.info(f"\nLandmark file structure: {structure}")
        struct_path = os.path.join(OUTPUT_DIR, "landmark_structure.json")
        with open(struct_path, "w") as f:
            json.dump(structure, f, indent=2)
    
    return gloss_index

if __name__ == "__main__":
    build_gloss_index()
