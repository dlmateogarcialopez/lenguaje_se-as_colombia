"""Actualiza index.json con fuente de cada seña, sin re-extraer landmarks."""
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))
SIGNS = os.path.join(BASE, "static", "signs")
VIDEOS = os.path.join(BASE, "static", "videos")

idx_path = os.path.join(SIGNS, "index.json")
with open(idx_path, encoding="utf-8") as f:
    indice = json.load(f)

videos_disponibles = set(os.listdir(VIDEOS)) if os.path.isdir(VIDEOS) else set()
nuevo = {}
for k, v in indice.items():
    label = v["label"] if isinstance(v, dict) else v
    video = v.get("video") if isinstance(v, dict) else None
    # Heuristica: si tiene video en /videos/, viene de LSCPROPIO; si no, de LSCS45
    fuente = "LSCPROPIO" if video else "LSCS45"
    nuevo[k] = {"label": label, "video": video, "source": fuente}

    # Actualizar tambien el JSON individual
    sign_path = os.path.join(SIGNS, k + ".json")
    if os.path.exists(sign_path):
        with open(sign_path, encoding="utf-8") as f:
            data = json.load(f)
        data["source"] = fuente
        with open(sign_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

with open(idx_path, "w", encoding="utf-8") as f:
    json.dump(nuevo, f, ensure_ascii=False, indent=2)
print(f"{len(nuevo)} senas reetiquetadas")
