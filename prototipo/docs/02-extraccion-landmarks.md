# 02 — Extracción de landmarks

## Qué son los landmarks

Puntos 3D normalizados que MediaPipe detecta en cada frame del video.

- **Pose**: 33 puntos del cuerpo (hombros, codos, muñecas, caderas, cabeza, etc.)
- **Hand**: 21 puntos por mano (muñeca + 4 puntos por dedo × 5 dedos)

Coordenadas: `x, y` en `[0, 1]` (normalizado al ancho/alto de imagen). `z` es profundidad relativa al sujeto, con valor mayor = más lejos de la cámara. **`z` es ruidoso** y se amortigua en el frontend.

### Índices clave de MediaPipe Pose

| ID  | Punto              |
|-----|--------------------|
| 0   | nariz              |
| 11  | hombro **izquierdo** |
| 12  | hombro derecho     |
| 13  | codo izquierdo     |
| 14  | codo derecho       |
| 15  | muñeca izquierda   |
| 16  | muñeca derecha     |
| 23  | cadera izquierda   |
| 24  | cadera derecha     |

⚠️ MediaPipe etiqueta "izquierda/derecha" desde el **punto de vista del sujeto** (no de la cámara).

### Índices clave de MediaPipe Hand (21 puntos)

```
       8   12  16  20
       |   |   |   |
       7   11  15  19
       |   |   |   |
       6   10  14  18    ← falanges distales / intermedias / proximales
       |   |   |   |
   4   5   9   13  17    ← nudillos (MCP)
   |
   3
   |
   2
   |
   1
   |
   0 ← muñeca
```

- `0` = muñeca
- `5, 9, 13, 17` = nudillos (índice, medio, anular, meñique) — base de cada dedo
- `1, 2, 3, 4` = pulgar

## Scripts de extracción

### `extraer.py` — para LSCS45 (JSON pre-procesado)

Lee `LSCS45/sample.json` que ya tiene landmarks. Solo reformatea al esquema interno.

### `extraer_videos.py` — para LSCPROPIO (videos crudos)

Pipeline completo:

```python
# 1. Configurar modelos MediaPipe (modo VIDEO para tracking temporal)
pose_opts = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="models/pose_landmarker_full.task"),
    running_mode=RunningMode.VIDEO)
hand_opts = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="models/hand_landmarker.task"),
    running_mode=RunningMode.VIDEO, num_hands=2)

# 2. Para cada video, recorrer frames
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)  # típicamente 30
ts = 0
while ret:
    ret, frame = cap.read()
    img = mp.Image(image_format=mp.ImageFormat.SRGB,
                   data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    pose_res = pose_lm.detect_for_video(img, ts)
    hand_res = hand_lm.detect_for_video(img, ts)
    # ...
    ts += int(1000 / fps)  # timestamp en milisegundos
```

### Asignación de manos en extracción

MediaPipe Hand devuelve hasta 2 detecciones con etiqueta `"Left"` / `"Right"`. **Esta etiqueta asume video en espejo y suele estar invertida** para grabaciones directas. Por eso:

- En extracción: se guardan ambas manos según la etiqueta original (sin corregir).
- En frontend: se reasignan por cercanía a las muñecas del cuerpo (ver `03-retargeting-avatar.md`).

## Esquema de salida

Cada `static/signs/<palabra>.json`:

```json
{
  "sign": "Lunes",
  "fps": 30.0,
  "video": "videos/lunes.m4v",
  "frames": [
    {
      "pose":   [[x0, x1, ..., x32], [y0, ..., y32], [z0, ..., z32]],
      "l_hand": [[x0, ..., x20],     [y0, ..., y20], [z0, ..., z20]],
      "r_hand": [[x0, ..., x20],     [y0, ..., y20], [z0, ..., z20]]
    },
    /* ... un objeto por frame ... */
  ]
}
```

### Decisiones de formato

- **Arrays separados por eje** (`[xs, ys, zs]`) en vez de objetos `{x, y, z}` → archivos ~30% más pequeños y fáciles de iterar en JS.
- **Si una mano no se detectó**: arrays vacíos `[[], [], []]`. El frontend ignora landmarks vacíos.
- **`fps`** guardado para reproducir a velocidad original.

## Índice general

`static/signs/index.json`:

```json
{
  "lunes":    { "label": "lunes",    "video": "videos/lunes.m4v" },
  "celular":  { "label": "celular",  "video": "videos/celular.mp4" },
  "bien":     "Bien"   // formato legacy de LSCS45 (sin video)
}
```

El frontend acepta ambos formatos: string (label sin video) u objeto.

## Cómo añadir más palabras

```bash
cd prototipo
# Editar extraer_videos.py si fuente cambia
python extraer_videos.py
```

Re-extrae todas las palabras de `LSCPROPIO/`. Cada video tarda ~5-15s.

## Pendiente: integrar LSC50

LSC50 ya tiene **landmarks CSV pre-extraídos** (`LSC50/LANDMARKS/`), un archivo por sample con columnas `landmark_N_x/y/z`. No necesita re-extracción con MediaPipe, solo un convertidor CSV → JSON con esquema interno. Ver `04-pendientes.md`.
