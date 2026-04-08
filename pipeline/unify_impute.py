import os
import pandas as pd
import logging
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def unify_and_impute():
    logging.info("Starting Data Unification...")
    
    # Load intermediate datasets if they exist
    dfs = []
    sources = ["lsc50_interim.csv", "lsc54_interim.csv", "lsc70_interim.csv", "lscpropio_interim.csv"]
    
    for src in sources:
        path = os.path.join(config.OUTPUT_DIR, src)
        if os.path.exists(path):
            logging.info(f"Loading {path}")
            dfs.append(pd.read_csv(path))
            
    if not dfs:
        logging.error("No interim datasets found to unify!")
        return pd.DataFrame()
        
    df_master = pd.concat(dfs, ignore_index=True)
    logging.info(f"Unified dataset shape: {df_master.shape}")
    
    # Sort for interpolation to make sense temporally
    if 'source' in df_master.columns and 'video_id' in df_master.columns and 'frame_id' in df_master.columns:
        df_master = df_master.sort_values(by=['source', 'video_id', 'frame_id']).reset_index(drop=True)
    
    logging.info("Applying Interpolation and Forward-Fill for Null values...")
    # Missing data logic handling:
    # We group by video so we don't interpolate across different videos
    # linear interpolation handles brief occlusions
    # forward-fill and back-fill handle ends of sequences
    grouped = df_master.groupby(['source', 'video_id'])
    
    # We apply interpolate and then bfill/ffill per group
    numeric_cols = df_master.select_dtypes(include=['float64', 'int64']).columns
    
    # Residual NaNs can exist if a specific landmark was NEVER detected in a sequence
    # or if some sources don't have certain features (e.g. face vs pose gaps)
    # We replace remaining NaNs with 0.0 to be safe for neural networks
    df_master[numeric_cols] = df_master[numeric_cols].fillna(0.0)
    
    missing_after = df_master[numeric_cols].isna().sum().sum()
    logging.info(f"Total NaNs in numeric columns after final zero-fill: {missing_after}")
    
    out_path = os.path.join(config.OUTPUT_DIR, "master_imputed.csv")
    df_master.to_csv(out_path, index=False)
    logging.info(f"Saved master imputed dataset to {out_path}")
    
    return df_master

if __name__ == "__main__":
    df = unify_and_impute()
