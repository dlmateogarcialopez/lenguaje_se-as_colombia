import os
import glob
import cv2
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
import mediapipe as mp
try:
    import mediapipe.solutions.holistic as mp_holistic
    import mediapipe.solutions.drawing_utils as mp_drawing
    import mediapipe.solutions.drawing_styles as mp_drawing_styles
except ImportError:
    import mediapipe.python.solutions.holistic as mp_holistic
    import mediapipe.python.solutions.drawing_utils as mp_drawing
    import mediapipe.python.solutions.drawing_styles as mp_drawing_styles

import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LSC70Converter:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.mp_holistic = mp_holistic
        self._visual_tested = False
        
    def process_sequence(self, sequence_dir):
        """Procesa una carpeta de imágenes estáticas (.jpg) de una sola seña/secuencia."""
        images = sorted(glob.glob(os.path.join(sequence_dir, "*.jpg")))
        if not images:
            return None
            
        vid_id = os.path.basename(sequence_dir)
        records = []
        
        with self.mp_holistic.Holistic(
            static_image_mode=False, # We use sequence processing
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=False
        ) as holistic:
            
            for frame_idx, img_path in enumerate(images):
                frame_data = {
                    'source': 'LSC70',
                    'video_id': vid_id,
                    'frame_id': frame_idx,
                    # We might not know the label directly from the folder name unless encoded.
                    # Commonly, folder name pattern "word_01" contains the label
                    'sign_label': vid_id.split('_')[0] if '_' in vid_id else vid_id
                }
                
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                        
                    # MediaPipe needs RGB
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    results = holistic.process(img_rgb)
                    
                    # Extraer Pose (33 landmarks)
                    if results.pose_landmarks:
                        for i, lm in enumerate(results.pose_landmarks.landmark):
                            frame_data[f'pose_{i}_x'] = lm.x
                            frame_data[f'pose_{i}_y'] = lm.y
                            frame_data[f'pose_{i}_z'] = lm.z
                    else:
                        # Rellenar con nans (opcional proactividad)
                        for i in range(33):
                            frame_data[f'pose_{i}_x'] = np.nan
                            frame_data[f'pose_{i}_y'] = np.nan
                            frame_data[f'pose_{i}_z'] = np.nan
                            
                    # Extraer Manos
                    # Left hand (21 landmarks)
                    if results.left_hand_landmarks:
                        for i, lm in enumerate(results.left_hand_landmarks.landmark):
                            frame_data[f'l_hand_{i}_x'] = lm.x
                            frame_data[f'l_hand_{i}_y'] = lm.y
                            frame_data[f'l_hand_{i}_z'] = lm.z
                    else:
                        for i in range(21):
                            frame_data[f'l_hand_{i}_x'] = np.nan
                            frame_data[f'l_hand_{i}_y'] = np.nan
                            frame_data[f'l_hand_{i}_z'] = np.nan
                            
                    # Right hand
                    if results.right_hand_landmarks:
                        for i, lm in enumerate(results.right_hand_landmarks.landmark):
                            frame_data[f'r_hand_{i}_x'] = lm.x
                            frame_data[f'r_hand_{i}_y'] = lm.y
                            frame_data[f'r_hand_{i}_z'] = lm.z
                    else:
                        for i in range(21):
                            frame_data[f'r_hand_{i}_x'] = np.nan
                            frame_data[f'r_hand_{i}_y'] = np.nan
                            frame_data[f'r_hand_{i}_z'] = np.nan
                            
                    # Face (468 landmarks)
                    if results.face_landmarks:
                        for i, lm in enumerate(results.face_landmarks.landmark):
                            frame_data[f'face_{i}_x'] = lm.x
                            frame_data[f'face_{i}_y'] = lm.y
                            frame_data[f'face_{i}_z'] = lm.z
                    else:
                        for i in range(468):
                            frame_data[f'face_{i}_x'] = np.nan
                            frame_data[f'face_{i}_y'] = np.nan
                            frame_data[f'face_{i}_z'] = np.nan
                            
                    # Testing visual validation save on first frame
                    if frame_idx == 0 and not self._visual_tested:
                        self.save_visual_test(img, results, vid_id)
                        self._visual_tested = True
                        
                    records.append(frame_data)
                    
                except Exception as e:
                    logging.warning(f"Error processing {img_path}: {e}")
                    
        return pd.DataFrame(records)

    def save_visual_test(self, img, results, vid_id):
        annotated_img = img.copy()
        mp_drawing.draw_landmarks(
            annotated_img, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
        mp_drawing.draw_landmarks(
            annotated_img, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS
        )
        mp_drawing.draw_landmarks(
            annotated_img, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS
        )
        out_path = os.path.join(config.OUTPUT_DIR, f"lsc70_validation_{vid_id}.jpg")
        cv2.imwrite(out_path, annotated_img)
        logging.info(f"Saved visual validation test to {out_path}")

    def process_all(self, limit=5):
        logging.info("Starting LSC70 Conversion...")
        # Recursively find all directories that contain .jpg files
        all_dirs = []
        for root, dirs, files in os.walk(self.base_dir):
            if any(f.lower().endswith('.jpg') for f in files):
                all_dirs.append(root)
        
        logging.info(f"Found {len(all_dirs)} sequence directories with images in LSC70.")
        
        all_dfs = []
        # Limit for demonstration
        for seq_dir in tqdm(all_dirs[:limit], desc="Converting LSC70 Images"):
            df_seq = self.process_sequence(seq_dir)
            if df_seq is not None and not df_seq.empty:
                all_dfs.append(df_seq)
                
        if not all_dfs:
            logging.warning("No data extracted from LSC70.")
            return pd.DataFrame()
            
        df_final = pd.concat(all_dfs, ignore_index=True)
        logging.info(f"Generated LSC70 dataset: {df_final.shape}")
        return df_final

if __name__ == "__main__":
    converter = LSC70Converter(config.LSC70_DIR)
    df_70 = converter.process_all(limit=5)
    
    if not df_70.empty:
        out_path = os.path.join(config.OUTPUT_DIR, "lsc70_interim.csv")
        df_70.to_csv(out_path, index=False)
        logging.info(f"Saved LSC70 interim data to {out_path}")
