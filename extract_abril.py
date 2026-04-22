import cv2
import pandas as pd
import numpy as np
import mediapipe.python.solutions.holistic as mp_holistic

video_file = r'D:\LSC\LSCPROPIO\abril\abril-persona1.mp4'
output_file = r'D:\LSC\pipeline\dynamic_landmarks\MES_ABRIL.csv'

cap = cv2.VideoCapture(video_file)
records = []
with mp_holistic.Holistic(
    static_image_mode=False, model_complexity=1,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
) as holistic:
    while True:
        ret, frame = cap.read()
        if not ret: break
        res = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        row = {}
        for idx, hm in enumerate([res.pose_landmarks, res.left_hand_landmarks, res.right_hand_landmarks, res.face_landmarks]):
            n_lms = [33, 21, 21, 468][idx]
            prefix = ['pose', 'l_hand', 'r_hand', 'face'][idx]
            if hm:
                for i, lm in enumerate(hm.landmark):
                    row[f'{prefix}_{i}_x'] = lm.x
                    row[f'{prefix}_{i}_y'] = lm.y
                    row[f'{prefix}_{i}_z'] = lm.z
            else:
                for i in range(n_lms):
                    row[f'{prefix}_{i}_x'] = np.nan; row[f'{prefix}_{i}_y'] = np.nan; row[f'{prefix}_{i}_z'] = np.nan
        records.append(row)

df = pd.DataFrame(records).fillna(0.0)
df.to_csv(output_file, index=True)
print("Done! Saved MES_ABRIL.csv")
