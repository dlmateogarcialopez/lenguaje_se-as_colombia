import json
import os
import shutil
import unicodedata

import cv2
import mediapipe as mp

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, "..", "LSCPROPIO")
OUT = os.path.join(BASE, "static", "signs")
OUT_VID = os.path.join(BASE, "static", "videos")
MODELS = os.path.join(BASE, "models")

BaseOptions = mp.tasks.BaseOptions
RunningMode = mp.tasks.vision.RunningMode
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions


def slug(name):
    nfkd = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in nfkd if not unicodedata.combining(c)).lower()
    return s.replace(" ", "")


def listar_videos():
    """{palabra: ruta_video} — prefiere persona1."""
    vids = {}
    for entrada in sorted(os.listdir(SRC)):
        ruta = os.path.join(SRC, entrada)
        if os.path.isdir(ruta):
            archivos = sorted(os.listdir(ruta))
            if archivos:
                vids[entrada] = os.path.join(ruta, archivos[0])
        elif entrada.lower().endswith((".mp4", ".m4v")):
            palabra = entrada.split("-persona")[0]
            vids.setdefault(palabra, ruta)
    return vids


def extraer(video_path, pose_lm, hand_lm):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames = []
    ts = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img = mp.Image(image_format=mp.ImageFormat.SRGB,
                       data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pose_res = pose_lm.detect_for_video(img, ts)
        hand_res = hand_lm.detect_for_video(img, ts)

        pose = [[], [], []]
        if pose_res.pose_landmarks:
            for lm in pose_res.pose_landmarks[0]:
                pose[0].append(lm.x); pose[1].append(lm.y); pose[2].append(lm.z)

        l_hand = [[], [], []]
        r_hand = [[], [], []]
        for i, handedness in enumerate(hand_res.handedness):
            destino = l_hand if handedness[0].category_name == "Left" else r_hand
            for lm in hand_res.hand_landmarks[i]:
                destino[0].append(lm.x); destino[1].append(lm.y); destino[2].append(lm.z)

        frames.append({"pose": pose, "l_hand": l_hand, "r_hand": r_hand})
        ts += int(1000 / fps)
    cap.release()
    return fps, frames


def main():
    os.makedirs(OUT_VID, exist_ok=True)
    idx_path = os.path.join(OUT, "index.json")
    index = {}
    if os.path.exists(idx_path):
        with open(idx_path, encoding="utf-8") as f:
            viejo = json.load(f)
        for k, v in viejo.items():
            index[k] = v if isinstance(v, dict) else {"label": v, "video": None}

    pose_opts = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.join(MODELS, "pose_landmarker_full.task")),
        running_mode=RunningMode.VIDEO)
    hand_opts = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.join(MODELS, "hand_landmarker.task")),
        running_mode=RunningMode.VIDEO, num_hands=2)

    vids = listar_videos()
    print(f"{len(vids)} palabras con video")
    for palabra, ruta in vids.items():
        nombre = slug(palabra)
        with PoseLandmarker.create_from_options(pose_opts) as pose_lm, \
             HandLandmarker.create_from_options(hand_opts) as hand_lm:
            fps, frames = extraer(ruta, pose_lm, hand_lm)
        ext = os.path.splitext(ruta)[1].lower()
        vid_dest = nombre + (".mp4" if ext == ".mp4" else ".m4v")
        shutil.copy(ruta, os.path.join(OUT_VID, vid_dest))
        with open(os.path.join(OUT, nombre + ".json"), "w", encoding="utf-8") as f:
            json.dump({"sign": palabra, "fps": fps, "video": "videos/" + vid_dest,
                       "frames": frames}, f)
        index[nombre] = {"label": palabra, "video": "videos/" + vid_dest}
        print(f"{palabra}: {len(frames)} frames @ {fps:.1f}fps")

    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)
    print("listo")


if __name__ == "__main__":
    main()
