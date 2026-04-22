import cv2, json, numpy as np
import mediapipe.python.solutions.holistic as mp_holistic

images = [
    r'C:\Users\DataLab\.gemini\antigravity\brain\654ad615-ca53-41f8-bd04-e53a48eacc2c\abril_real_01.png',
    r'C:\Users\DataLab\.gemini\antigravity\brain\654ad615-ca53-41f8-bd04-e53a48eacc2c\abril_real_02.png',
    r'C:\Users\DataLab\.gemini\antigravity\brain\654ad615-ca53-41f8-bd04-e53a48eacc2c\abril_real_03.png'
]

results = []
with mp_holistic.Holistic(static_image_mode=True, model_complexity=0) as holistic:
    for path in images:
        img = cv2.imread(path)
        if img is None: continue
        res = holistic.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        row = []
        for hm, num_pts in [(res.pose_landmarks, 33), (res.left_hand_landmarks, 21), (res.right_hand_landmarks, 21), (res.face_landmarks, 468)]:
            if hm:
                for lm in hm.landmark: row.extend([lm.x, lm.y, lm.z])
            else:
                row.extend([0.0]*3*num_pts)
        results.append(row)

arr = np.array(results)
print("Extracted shape:", arr.shape)
np.save(r'd:\LSC\pipeline\dynamic_landmarks\abril_key.npy', arr)
print("Saved npy!")
