import os
import glob
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
import ijson
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LSC50Ingestor:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        
    def process(self):
        logging.info("Starting LSC50 Ingestion...")
        body_dir = os.path.join(self.base_dir, "BODY_LANDMARKS")
        lhand_dir = os.path.join(self.base_dir, "HANDS_LANDMARKS", "LEFT_HAND_LANDMARKS")
        rhand_dir = os.path.join(self.base_dir, "HANDS_LANDMARKS", "RIGHT_HAND_LANDMARKS")
        face_dir = os.path.join(self.base_dir, "FACE_LANDMARKS")
        
        body_files = glob.glob(os.path.join(body_dir, "*.csv"))
        
        # Remove sample slicing to process ALL files eventually, but let's process all now
        all_frames = []
        for bf in tqdm(body_files, desc="Processing LSC50"):
            vid_id = os.path.basename(bf).replace('.csv', '')
            try:
                df_body = pd.read_csv(bf)
                # Map Body (LSC50) to pose (MediaPipe naming)
                pose_map = {f"landmark_{i}_{ax}": f"pose_{i}_{ax}" for i in range(33) for ax in ['x','y','z']}
                df_body.rename(columns=pose_map, inplace=True)
                
                df_body['video_id'] = vid_id
                df_body['source'] = 'LSC50'
                df_body['frame_id'] = df_body.index 
                
                # Check LEFT HAND
                lhf = os.path.join(lhand_dir, f"{vid_id}.csv")
                if os.path.exists(lhf):
                    df_lh = pd.read_csv(lhf)
                    df_lh.rename(columns={f"landmark_{i}_{ax}": f"l_hand_{i}_{ax}" for i in range(21) for ax in ['x','y','z']}, inplace=True)
                    df_body = df_body.join(df_lh)
                
                # Check RIGHT HAND
                rhf = os.path.join(rhand_dir, f"{vid_id}.csv")
                if os.path.exists(rhf):
                    df_rh = pd.read_csv(rhf)
                    df_rh.rename(columns={f"landmark_{i}_{ax}": f"r_hand_{i}_{ax}" for i in range(21) for ax in ['x','y','z']}, inplace=True)
                    df_body = df_body.join(df_rh)
                    
                # Check FACE
                ff = os.path.join(face_dir, f"{vid_id}.csv")
                if os.path.exists(ff):
                    df_face = pd.read_csv(ff)
                    df_face.rename(columns={f"landmark_{i}_{ax}": f"face_{i}_{ax}" for i in range(468) for ax in ['x','y','z']}, inplace=True)
                    df_body = df_body.join(df_face)
                    
                all_frames.append(df_body)
            except Exception as e:
                logging.error(f"Failed to read {bf}: {e}")
            
        if not all_frames:
            logging.warning("No files found for LSC50.")
            return pd.DataFrame()
            
        df_final = pd.concat(all_frames, ignore_index=True)
        logging.info(f"Loaded LSC50: {df_final.shape}")
        return df_final

class LSC54Ingestor:
    def __init__(self, json_path, out_csv_path=None):
        self.json_path = json_path
        self.out_csv_path = out_csv_path or os.path.join(config.OUTPUT_DIR, "lsc54_interim.csv")
        
    def process(self):
        logging.info(f"Starting LSC54 (LSCS45) Ingestion from {self.json_path}...")
        records = []
        chunk_size = 50000
        first_chunk = True
        
        if os.path.exists(self.out_csv_path):
            os.remove(self.out_csv_path)

        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                parser = ijson.parse(f)
                frame_data = {}
                count = 0
                
                for prefix, event, value in parser:
                    
                    parts = prefix.split('.')
                    if len(parts) >= 6:
                        signer, topic, sign, video, rep, frame = parts[:6]
                        
                        if event == 'start_map' and len(parts) == 6:
                            frame_data = {
                                'source': 'LSC54',
                                'signer': signer,
                                'topic': topic,
                                'sign_label': sign,
                                'video_id': video,
                                'repetition': rep,
                                'frame_id': frame
                            }
                        elif event == 'number':
                            if len(parts) == 9 and parts[8] == 'item':
                                part = parts[6] 
                                axis = parts[7] 
                                
                                key_prefix = f"{part}_{axis}_list"
                                if key_prefix not in frame_data:
                                    frame_data[key_prefix] = []
                                frame_data[key_prefix].append(value)
                                
                        elif event == 'end_map' and len(parts) == 6:
                            flat_frame = frame_data.copy()
                            for k in list(flat_frame.keys()):
                                if k.endswith('_list'):
                                    vals = flat_frame.pop(k)
                                    part_axis = k.replace('_list', '') 
                                    part, axis = part_axis.rsplit('_', 1) 
                                    for i, v in enumerate(vals):
                                        flat_frame[f"{part}_{i}_{axis}"] = v
                            
                            records.append(flat_frame)
                            count += 1
                            if count % chunk_size == 0:
                                df_chunk = pd.DataFrame(records)
                                df_chunk.to_csv(self.out_csv_path, mode='a', index=False, header=first_chunk)
                                first_chunk = False
                                records = []
                                logging.info(f"Parsed and dumped {count} frames from LSC54 to disk...")
                                
                if records:
                    df_chunk = pd.DataFrame(records)
                    df_chunk.to_csv(self.out_csv_path, mode='a', index=False, header=first_chunk)
                    logging.info(f"Parsed and dumped remaining {len(records)} frames. Total: {count}")
                    records = []
                    
        except Exception as e:
            logging.error(f"Error parsing LSC54: {e}")
            
        logging.info(f"Finished LSC54 ingestion. Saved to {self.out_csv_path}")
        return pd.DataFrame()

if __name__ == "__main__":
    lsc50 = LSC50Ingestor(config.LSC50_LANDMARKS_DIR)
    df_50 = lsc50.process()
    
    lsc54 = LSC54Ingestor(config.LSC54_SAMPLE_JSON_PATH)
    df_54 = lsc54.process()
    
    if not df_50.empty:
        out_path_50 = os.path.join(config.OUTPUT_DIR, "lsc50_interim.csv")
        df_50.to_csv(out_path_50, index=False)
        logging.info(f"Saved LSC50 interim data to {out_path_50}")

    if not df_54.empty:
        out_path_54 = os.path.join(config.OUTPUT_DIR, "lsc54_interim.csv")
        df_54.to_csv(out_path_54, index=False)
        logging.info(f"Saved LSC54 interim data to {out_path_54}")
