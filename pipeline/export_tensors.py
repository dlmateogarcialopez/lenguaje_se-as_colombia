import os
import pandas as pd
import numpy as np
import logging
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def export_tensors(sequence_length=60):
    logging.info(f"Starting Tensor Export with Fixed Sequence Length T={sequence_length}...")
    
    in_path = os.path.join(config.OUTPUT_DIR, "master_normalized.csv")
    if not os.path.exists(in_path):
        logging.error(f"Normalized dataset not found at {in_path}")
        return
        
    df = pd.read_csv(in_path)
    
    # We group by source and video_id to get sequences
    # Some older datasets might only have 'video_id', some have both.
    if 'source' in df.columns and 'video_id' in df.columns:
        grouped = df.groupby(['source', 'video_id'])
    else:
        grouped = df.groupby(['video_id'])
        
    X = []
    y = []
    
    # Identify feature columns (everything that isn't metadata) strictly by MediaPipe Holistic standards
    # Total exactly 1629 columns (543 landmarks * 3 coordinates)
    feat_cols = []
    
    # 33 Pose landmarks
    for i in range(33): feat_cols.extend([f"pose_{i}_x", f"pose_{i}_y", f"pose_{i}_z"])
    # 21 Left Hand landmarks
    for i in range(21): feat_cols.extend([f"l_hand_{i}_x", f"l_hand_{i}_y", f"l_hand_{i}_z"])
    # 21 Right Hand landmarks
    for i in range(21): feat_cols.extend([f"r_hand_{i}_x", f"r_hand_{i}_y", f"r_hand_{i}_z"])
    # 468 Face landmarks
    for i in range(468): feat_cols.extend([f"face_{i}_x", f"face_{i}_y", f"face_{i}_z"])

    # Discard anything else (like internal indices 'Unnamed: 0', 'landmark_i_x' raw, etc.)
    logging.info(f"Feature Vector Dimension per frame: {len(feat_cols)}")
    
    # Ensure frame_id is numeric for sorting
    df['frame_id'] = pd.to_numeric(df['frame_id'], errors='coerce').fillna(0).astype(int)
    
    for (source, vid), group in grouped:
        # Sort group by frame_id
        group = group.sort_values(by='frame_id')
        features = group[feat_cols].values # Shape: (frames_in_video, num_features)
        
        # Get label
        label = group['sign_label'].iloc[0] if 'sign_label' in group.columns else vid
        
        # Pad or Truncate to sequence_length (T)
        frames_in_video = features.shape[0]
        
        if frames_in_video >= sequence_length:
            # Uniform Sampling / Truncation
            # Here we just simple truncate for pipeline example, or uniform sampling
            indices = np.linspace(0, frames_in_video - 1, sequence_length, dtype=int)
            seq = features[indices]
        else:
            # Post-Padding with zeros
            pad_len = sequence_length - frames_in_video
            padding = np.zeros((pad_len, features.shape[1]))
            seq = np.vstack([features, padding])
            
        X.append(seq)
        y.append(label)
        
    X_tensor = np.array(X)
    y_labels = np.array(y)
    
    logging.info(f"Final Tensor Shape (X): {X_tensor.shape}")
    logging.info(f"Labels Shape (y): {y_labels.shape}")
    
    # Export
    np.save(os.path.join(config.OUTPUT_DIR, "LSC_dataset_unificado.npy"), X_tensor)
    np.save(os.path.join(config.OUTPUT_DIR, "LSC_labels.npy"), y_labels)
    
    logging.info("Tensors successfully exported!")
    
if __name__ == "__main__":
    export_tensors(sequence_length=60)
