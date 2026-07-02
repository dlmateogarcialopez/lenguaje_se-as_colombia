"""Integra LSC50: lee landmarks CSV (ya extraidos por el dataset),
convierte avi -> mp4 con ffmpeg, escribe JSON de seña con esquema interno."""
import csv
import json
import os
import subprocess
import unicodedata

import openpyxl

BASE = os.path.dirname(os.path.abspath(__file__))
LSC50 = os.path.join(BASE, "..", "LSC50")
TIMESTAMPS = os.path.join(LSC50, "IMU", "INFO", "Timestamps.xlsx")
LM_BODY = os.path.join(LSC50, "LANDMARKS", "BODY_LANDMARKS")
LM_LEFT = os.path.join(LSC50, "LANDMARKS", "HANDS_LANDMARKS", "LEFT_HAND_LANDMARKS")
LM_RIGHT = os.path.join(LSC50, "LANDMARKS", "HANDS_LANDMARKS", "RIGHT_HAND_LANDMARKS")
LM_FACE = os.path.join(LSC50, "LANDMARKS", "FACE_LANDMARKS")
VIDEOS_AVI = os.path.join(LSC50, "VIDEOS", "COLOR_BODY")

# Indices del FaceMesh (468) que conserva el frontend (cejas, ojos, boca).
# Debe coincidir con FACE_INDICES de extraer_videos.py / watcher.py.
FACE_INDICES = [
    70, 63, 105, 66, 107,
    336, 296, 334, 293, 300,
    159, 145, 33, 133,
    386, 374, 263, 362,
    13, 14, 78, 308,
    17, 0,
    61, 291,
]

OUT_SIGNS = os.path.join(BASE, "static", "signs")
OUT_VIDEOS = os.path.join(BASE, "static", "videos")

# Elegir un signante por defecto (algunos signantes tienen calidad variable).
SIGNER_DEFAULT = "0000"
REP_DEFAULT = "0000"
FPS_LSC50 = 30  # estimado, no esta documentado explicitamente


def slug(s):
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace(" ", "").replace("?", "").replace("¿", "")
    return "".join(c for c in s if c.isalnum())


def leer_nombres():
    """sign_id (4 digitos) -> nombre legible."""
    wb = openpyxl.load_workbook(TIMESTAMPS, read_only=True)
    ws = wb["Volunteer 1"]
    nombres = {}
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        if i == 0:
            continue
        sign_id, nombre = row[0], row[1]
        nombres[sign_id] = nombre.strip()
    return nombres


def leer_csv(path):
    """CSV con header 'landmark_N_x/y/z' -> [[xs], [ys], [zs]] por frame."""
    if not os.path.exists(path):
        return []
    frames = []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            vals = [float(v) for v in row[1:]]  # primera col = indice de frame
            xs = vals[0::3]
            ys = vals[1::3]
            zs = vals[2::3]
            frames.append([xs, ys, zs])
    return frames


def convertir_video(avi_path, mp4_path):
    if os.path.exists(mp4_path):
        return
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", avi_path,
         "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-an", mp4_path],
        check=True,
    )


def main():
    os.makedirs(OUT_SIGNS, exist_ok=True)
    os.makedirs(OUT_VIDEOS, exist_ok=True)

    nombres = leer_nombres()
    print(f"{len(nombres)} senas en Timestamps")

    idx_path = os.path.join(OUT_SIGNS, "index.json")
    indice = {}
    if os.path.exists(idx_path):
        with open(idx_path, encoding="utf-8") as f:
            indice = json.load(f)

    procesadas = 0
    for sign_id, nombre in nombres.items():
        base = f"{sign_id}_{SIGNER_DEFAULT}_{REP_DEFAULT}"
        pose = leer_csv(os.path.join(LM_BODY, base + ".csv"))
        l_hand = leer_csv(os.path.join(LM_LEFT, base + ".csv"))
        r_hand = leer_csv(os.path.join(LM_RIGHT, base + ".csv"))
        face_full = leer_csv(os.path.join(LM_FACE, base + ".csv"))
        avi = os.path.join(VIDEOS_AVI, base + ".avi")

        if not pose or not os.path.exists(avi):
            print(f"[!] {nombre} ({sign_id}): falta CSV o video, omitido")
            continue

        n_frames = len(pose)
        frames = []
        for i in range(n_frames):
            face = [[], [], []]
            if i < len(face_full) and len(face_full[i][0]) >= max(FACE_INDICES) + 1:
                for idx in FACE_INDICES:
                    face[0].append(face_full[i][0][idx])
                    face[1].append(face_full[i][1][idx])
                    face[2].append(face_full[i][2][idx])
            frames.append({
                "pose": pose[i],
                "l_hand": l_hand[i] if i < len(l_hand) else [[], [], []],
                "r_hand": r_hand[i] if i < len(r_hand) else [[], [], []],
                "face": face,
            })

        clave = slug(nombre)
        mp4_dest = clave + ".mp4"
        try:
            convertir_video(avi, os.path.join(OUT_VIDEOS, mp4_dest))
        except subprocess.CalledProcessError as e:
            print(f"[!] {nombre}: ffmpeg fallo, omitido")
            continue

        with open(os.path.join(OUT_SIGNS, clave + ".json"), "w", encoding="utf-8") as f:
            json.dump({"sign": nombre, "fps": FPS_LSC50, "source": "LSC50",
                       "video": "videos/" + mp4_dest, "frames": frames}, f)
        indice[clave] = {"label": nombre, "video": "videos/" + mp4_dest, "source": "LSC50"}
        procesadas += 1
        print(f"  {nombre}: {n_frames} frames")

    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(indice, f, ensure_ascii=False, indent=2)
    print(f"listo: {procesadas} senas integradas")


if __name__ == "__main__":
    main()
