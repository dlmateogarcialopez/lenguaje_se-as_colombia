import os
import pandas as pd
import numpy as np
import logging
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def normalize_spatial(df):
    logging.info("Starting Spatial Normalization...")
    # Point of origin will be the wrist (landmark 0) of the right hand and left hand respectively.
    # Since LSC involves both hands, we might center the whole pose to pose landmark 0 (nose)
    # or follow instructions: "muñeca (Hand Landmark 0) como punto de origen (0,0,0)"
    
    # Identify wrist columns
    r_wrist_x = 'r_hand_0_x'
    r_wrist_y = 'r_hand_0_y'
    r_wrist_z = 'r_hand_0_z'
    
    l_wrist_x = 'l_hand_0_x'
    l_wrist_y = 'l_hand_0_y'
    l_wrist_z = 'l_hand_0_z'
    
    # Process Right Hand
    if r_wrist_x in df.columns:
        for i in range(21):
            if f'r_hand_{i}_x' in df.columns:
                df[f'r_hand_{i}_x'] -= df[r_wrist_x]
                df[f'r_hand_{i}_y'] -= df[r_wrist_y]
                df[f'r_hand_{i}_z'] -= df[r_wrist_z]
                
    # Process Left Hand
    if l_wrist_x in df.columns:
        for i in range(21):
            if f'l_hand_{i}_x' in df.columns:
                df[f'l_hand_{i}_x'] -= df[l_wrist_x]
                df[f'l_hand_{i}_y'] -= df[l_wrist_y]
                df[f'l_hand_{i}_z'] -= df[l_wrist_z]
                
    # Centering pose and face using midpoint of shoulders (pose_11 and pose_12)
    if 'pose_11_x' in df.columns and 'pose_12_x' in df.columns:
        mid_x = (df['pose_11_x'] + df['pose_12_x']) / 2.0
        mid_y = (df['pose_11_y'] + df['pose_12_y']) / 2.0
        mid_z = (df['pose_11_z'] + df['pose_12_z']) / 2.0
        
        for i in range(33):
            if f'pose_{i}_x' in df.columns:
                df[f'pose_{i}_x'] -= mid_x
                df[f'pose_{i}_y'] -= mid_y
                df[f'pose_{i}_z'] -= mid_z
                
        for i in range(468):
            if f'face_{i}_x' in df.columns:
                df[f'face_{i}_x'] -= mid_x
                df[f'face_{i}_y'] -= mid_y
                df[f'face_{i}_z'] -= mid_z
                
    # Check normalizations works (approximations to 0 due to floating point math)
    if r_wrist_x in df.columns:
        assert np.isclose(df[r_wrist_x].mean(), 0.0, atol=1e-5), "Right Wrist X is not normalized"
        
    logging.info("Spatial Normalization completed successfully.")
    
    out_path = os.path.join(config.OUTPUT_DIR, "master_normalized.csv")
    df.to_csv(out_path, index=False)
    logging.info(f"Saved normalized dataset to {out_path}")
    return df

if __name__ == "__main__":
    in_path = os.path.join(config.OUTPUT_DIR, "master_imputed.csv")
    if os.path.exists(in_path):
        df_imputed = pd.read_csv(in_path)
        normalize_spatial(df_imputed)
    else:
        logging.error(f"Input file not found at {in_path}. Run unification step first.")
