import os
import glob
import pandas as pd
import numpy as np
import cv2
import sys
from tqdm import tqdm

# Add current dir to path to import config and convert_lsc70
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from convert_lsc70 import LSC70Converter

DYNAMIC_DIR = os.path.join(config.OUTPUT_DIR, "dynamic_landmarks")
os.makedirs(DYNAMIC_DIR, exist_ok=True)

# Mapping for digits that are missing in LSC70AN but can be mapped to letters
DIGIT_MAPPING = {
    "0": "O",
    "2": "V",
    "3": "W"
}

def extract_all():
    converter = LSC70Converter(config.LSC70_DIR)
    root_an = os.path.join(config.LSC70_DIR, "LSC70AN")
    
    # We only need one person's data for consistency (Per01)
    # But some letters might be better in others? We'll stick to Per01 first
    signer = "Per01"
    signer_path = os.path.join(root_an, signer)
    
    if not os.path.exists(signer_path):
        print(f"Signer path not found: {signer_path}")
        return

    folders = os.listdir(signer_path)
    print(f"Extracting landmarks for {len(folders)} alphanumeric folders form {signer}...")

    for folder in tqdm(folders):
        folder_path = os.path.join(signer_path, folder)
        if not os.path.isdir(folder_path):
            continue
            
        # Extract landmarks (taking 5-10 frames for some variation)
        df = converter.process_sequence(folder_path)
        if df is not None and not df.empty:
            # Save to dynamic landmarks dir
            out_file = os.path.join(DYNAMIC_DIR, f"LETRA_{folder}.csv") if folder.isalpha() else os.path.join(DYNAMIC_DIR, f"NUMERO_{folder}.csv")
            df.to_csv(out_file, index=False)

    # Handle mapped digits
    for digit, letter in DIGIT_MAPPING.items():
        src = os.path.join(DYNAMIC_DIR, f"LETRA_{letter}.csv")
        if os.path.exists(src):
            dst = os.path.join(DYNAMIC_DIR, f"NUMERO_{digit}.csv")
            df_digit = pd.read_csv(src)
            # Update labels inside
            df_digit['sign_label'] = f"NUMERO_{digit}"
            df_digit.to_csv(dst, index=False)
            print(f"Mapped {digit} to {letter}")

if __name__ == "__main__":
    extract_all()
