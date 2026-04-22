import os
import cv2
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ingest_lscpropio import LSCPropioIngestor
import config

out_dir = os.path.join(config.OUTPUT_DIR, "dynamic_landmarks")
os.makedirs(out_dir, exist_ok=True)

ingestor = LSCPropioIngestor(base_dir=config.LSCPROPIO_DIR, frame_step=1)
videos = ingestor.discover_videos()

# Filter only months (or just process everything we can quickly)
months = ['enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio', 'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre']

found = 0
for v_path, sign_label, signer_id in videos:
    if sign_label.lower() in months:
        print(f"Extracting {sign_label} from {v_path}...")
        records = ingestor._extract_video_landmarks(v_path, sign_label, signer_id)
        if records:
            df = pd.DataFrame(records)
            # The ingestor outputs standard columns: 'pose_0_x', etc.
            # But load_dynamic_csv in motion_synthesizer.py expects NO headers, just the 1629 floats!
            # Wait! load_dynamic_csv logic in motion_synthesizer.py just skips line 0.
            # And it expects: 0, x, y, z, ... 
            
            # Reconstruct the 1629 columns EXACTLY as load_dynamic_csv expects:
            # We must output columns matching MediaPipe flat array:
            out_rows = []
            for r in records:
                flat = []
                for i in range(33):
                    flat.extend([r.get(f'pose_{i}_x', 0), r.get(f'pose_{i}_y', 0), r.get(f'pose_{i}_z', 0)])
                for i in range(21):
                    flat.extend([r.get(f'l_hand_{i}_x', 0), r.get(f'l_hand_{i}_y', 0), r.get(f'l_hand_{i}_z', 0)])
                for i in range(21):
                    flat.extend([r.get(f'r_hand_{i}_x', 0), r.get(f'r_hand_{i}_y', 0), r.get(f'r_hand_{i}_z', 0)])
                for i in range(468):
                    flat.extend([r.get(f'face_{i}_x', 0), r.get(f'face_{i}_y', 0), r.get(f'face_{i}_z', 0)])
                
                # Prepend the frame index format. load_dynamic_csv ignores part 0.
                out_rows.append([0] + flat)
            
            out_df = pd.DataFrame(out_rows)
            # Add a fake header to skip 
            out_df.columns = ['id'] + [f'val_{i}' for i in range(len(flat))]
            
            out_path = os.path.join(out_dir, f"MES_{sign_label.upper()}.csv")
            out_df.to_csv(out_path, index=False)
            print(f"Saved {len(out_df)} frames to {out_path}.")
            found += 1

print(f"Completed extracting {found} month videos.")
