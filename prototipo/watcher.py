"""Vigila LSCPROPIO/ y extrae landmarks automaticamente cuando aparece un video nuevo.

Convencion:
  - Crear una carpeta con el nombre de la palabra en LSCPROPIO/
  - Poner el video dentro (cualquier nombre .mp4 o .m4v)
  - El watcher lo detecta, extrae landmarks + face, actualiza index.json

Uso:
  python watcher.py
"""
import json
import os
import shutil
import time
import traceback
import unicodedata

import cv2
import mediapipe as mp

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, "..", "LSCPROPIO")
OUT = os.path.join(BASE, "static", "signs")
OUT_VID = os.path.join(BASE, "static", "videos")
MODELS = os.path.join(BASE, "models")
INTERVALO = 3  # segundos entre escaneos
ESTABLE_MS = 1500  # ignorar archivos que se modificaron hace menos (todavia copiando)

BaseOptions = mp.tasks.BaseOptions
RunningMode = mp.tasks.vision.RunningMode
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

FACE_INDICES = [
    70, 63, 105, 66, 107,
    336, 296, 334, 293, 300,
    159, 145, 33, 133,
    386, 374, 263, 362,
    13, 14, 78, 308,
    17, 0,
    61, 291,
]


def slug(name):
    nfkd = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in nfkd if not unicodedata.combining(c)).lower()
    return "".join(c for c in s if c.isalnum())


def cargar_indice():
    p = os.path.join(OUT, "index.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


def guardar_indice(idx):
    with open(os.path.join(OUT, "index.json"), "w", encoding="utf-8") as f:
        json.dump(idx, f, ensure_ascii=False, indent=2)


def listar_pendientes(indice):
    """Devuelve [(palabra, ruta_video, clave)] de videos que aun no estan en el indice."""
    pendientes = []
    if not os.path.isdir(SRC):
        return pendientes
    for entrada in sorted(os.listdir(SRC)):
        # Ignorar carpetas de trabajo/staging
        if entrada.startswith(("tmp", "_", ".")):
            continue
        ruta = os.path.join(SRC, entrada)
        if os.path.isdir(ruta):
            archivos = sorted(
                a for a in os.listdir(ruta)
                if a.lower().endswith((".mp4", ".m4v", ".avi", ".mov"))
            )
            if archivos:
                video = os.path.join(ruta, archivos[0])
                palabra = entrada
        elif entrada.lower().endswith((".mp4", ".m4v", ".avi", ".mov")):
            video = ruta
            palabra = entrada.rsplit("-persona", 1)[0].rsplit(".", 1)[0]
        else:
            continue

        clave = slug(palabra)
        if not clave or clave in indice:
            continue

        # ignorar archivos modificados hace muy poco (todavia se estan copiando)
        edad_ms = (time.time() - os.path.getmtime(video)) * 1000
        if edad_ms < ESTABLE_MS:
            continue

        pendientes.append((palabra, video, clave))
    return pendientes


def extraer_landmarks(video_path, pose_lm, hand_lm, face_lm):
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
        face_res = face_lm.detect_for_video(img, ts)

        pose = [[], [], []]
        pose_world = [[], [], []]
        if pose_res.pose_landmarks:
            for lm in pose_res.pose_landmarks[0]:
                pose[0].append(lm.x); pose[1].append(lm.y); pose[2].append(lm.z)
        if pose_res.pose_world_landmarks:
            for lm in pose_res.pose_world_landmarks[0]:
                pose_world[0].append(lm.x); pose_world[1].append(lm.y); pose_world[2].append(lm.z)

        l_hand = [[], [], []]
        r_hand = [[], [], []]
        for i, handedness in enumerate(hand_res.handedness):
            destino = l_hand if handedness[0].category_name == "Left" else r_hand
            for lm in hand_res.hand_landmarks[i]:
                destino[0].append(lm.x); destino[1].append(lm.y); destino[2].append(lm.z)

        face = [[], [], []]
        if face_res.face_landmarks:
            lms = face_res.face_landmarks[0]
            for idx in FACE_INDICES:
                lm = lms[idx]
                face[0].append(lm.x); face[1].append(lm.y); face[2].append(lm.z)

        frames.append({"pose": pose, "pose_world": pose_world,
                       "l_hand": l_hand, "r_hand": r_hand, "face": face})
        ts += int(1000 / fps)
    cap.release()
    return fps, frames


def convertir_a_mp4(src, dst):
    """Convierte cualquier formato a mp4 servible en navegador."""
    import subprocess
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", src,
         "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-an", dst],
        check=True,
    )


def procesar(palabra, video_path, clave):
    print(f"[+] procesando '{palabra}' ({os.path.basename(video_path)})...")
    # Crear modelos NUEVOS por cada video (MediaPipe en modo VIDEO acumula
    # timestamps y exige monotonicidad; reusarlos entre archivos falla).
    pose_lm, hand_lm, face_lm = crear_modelos()
    try:
        fps, frames = extraer_landmarks(video_path, pose_lm, hand_lm, face_lm)
    finally:
        pose_lm.close(); hand_lm.close(); face_lm.close()
    if not frames:
        print(f"[!] {palabra}: video vacio, omitido")
        return None

    ext = os.path.splitext(video_path)[1].lower()
    vid_dest_name = clave + ".mp4"
    vid_dest_path = os.path.join(OUT_VID, vid_dest_name)
    if ext == ".mp4":
        shutil.copy(video_path, vid_dest_path)
    else:
        convertir_a_mp4(video_path, vid_dest_path)

    with open(os.path.join(OUT, clave + ".json"), "w", encoding="utf-8") as f:
        json.dump({"sign": palabra, "fps": fps, "source": "LSCPROPIO",
                   "video": "videos/" + vid_dest_name, "frames": frames}, f)

    print(f"    {len(frames)} frames @ {fps:.1f}fps -> {clave}.json")
    return {"label": palabra, "video": "videos/" + vid_dest_name, "source": "LSCPROPIO"}


def crear_modelos():
    pose_opts = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.join(MODELS, "pose_landmarker_full.task")),
        running_mode=RunningMode.VIDEO)
    hand_opts = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.join(MODELS, "hand_landmarker.task")),
        running_mode=RunningMode.VIDEO, num_hands=2)
    face_opts = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.join(MODELS, "face_landmarker.task")),
        running_mode=RunningMode.VIDEO)
    return (
        PoseLandmarker.create_from_options(pose_opts),
        HandLandmarker.create_from_options(hand_opts),
        FaceLandmarker.create_from_options(face_opts),
    )


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(OUT_VID, exist_ok=True)
    print(f"[watcher] vigilando {SRC}")
    print(f"[watcher] cada {INTERVALO}s · ignora archivos < {ESTABLE_MS}ms de edad")
    while True:
        try:
            indice = cargar_indice()
            pendientes = listar_pendientes(indice)
            for palabra, video, clave in pendientes:
                try:
                    entrada = procesar(palabra, video, clave)
                    if entrada:
                        indice[clave] = entrada
                        guardar_indice(indice)
                except Exception:
                    print(f"[!] error procesando {palabra}:")
                    traceback.print_exc()
        except Exception:
            traceback.print_exc()
        time.sleep(INTERVALO)


if __name__ == "__main__":
    main()
