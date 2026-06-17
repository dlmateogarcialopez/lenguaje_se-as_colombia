# 04 — Pendientes y mejoras

## Integrar LSC50

LSC50 tiene 50 señas × 5 signantes × 4 repeticiones = **1000 muestras** con landmarks CSV ya extraídos. No requiere MediaPipe, solo conversión de formato.

### Estructura LSC50

```
LSC50/
├── VIDEOS/
│   └── COLOR_BODY/<SIGN>_<SIGNER>_<REP>.avi   (1000 archivos)
└── LANDMARKS/
    ├── BODY_LANDMARKS/<SIGN>_<SIGNER>_<REP>.csv     (33 puntos pose)
    ├── HANDS_LANDMARKS/
    │   ├── LEFT_HAND_LANDMARKS/<...>.csv            (21 puntos)
    │   └── RIGHT_HAND_LANDMARKS/<...>.csv           (21 puntos)
    └── FACE_LANDMARKS/<...>.csv                     (468 puntos, no usado todavía)
```

### Formato CSV

```csv
,landmark_0_x,landmark_0_y,landmark_0_z,landmark_1_x,...
0,0.5210,0.4341,-0.2906,0.5279,...
1,0.5210,0.4347,-0.3025,0.5279,...
```

Cada fila = un frame. Columnas: `landmark_N_x/y/z` para N = 0..32 (pose) o 0..20 (mano).

### Script propuesto: `extraer_lsc50.py`

Pseudo-código:

```python
import os, json, csv, shutil
from collections import defaultdict

BASE = "LSC50"
NOMBRES_SENAS = {  # mapeo 0000 → "Hola", 0001 → "Bien", etc.
    "0000": "Hola",
    "0001": "Bien",
    # ...completar 50 desde alguna fuente o README
}

def leer_csv_landmarks(path, n_puntos):
    """Devuelve [[xs], [ys], [zs]] por frame."""
    frames = []
    with open(path) as f:
        reader = csv.reader(f); next(reader)  # skip header
        for row in reader:
            vals = [float(x) for x in row[1:]]  # ignora primera columna (índice)
            xs = vals[0::3]; ys = vals[1::3]; zs = vals[2::3]
            frames.append([xs, ys, zs])
    return frames

# Agrupar por seña, elegir una muestra (signer_0, rep_0)
for sign_id, nombre in NOMBRES_SENAS.items():
    pose   = leer_csv_landmarks(f"{BASE}/LANDMARKS/BODY_LANDMARKS/{sign_id}_0000_0000.csv", 33)
    l_hand = leer_csv_landmarks(f"{BASE}/LANDMARKS/HANDS_LANDMARKS/LEFT_HAND_LANDMARKS/{sign_id}_0000_0000.csv", 21)
    r_hand = leer_csv_landmarks(f"{BASE}/LANDMARKS/HANDS_LANDMARKS/RIGHT_HAND_LANDMARKS/{sign_id}_0000_0000.csv", 21)

    # Combinar por frame
    frames = []
    for i in range(len(pose)):
        frames.append({
            "pose": pose[i],
            "l_hand": l_hand[i] if i < len(l_hand) else [[],[],[]],
            "r_hand": r_hand[i] if i < len(r_hand) else [[],[],[]],
        })

    # Copiar video y guardar JSON
    shutil.copy(f"{BASE}/VIDEOS/COLOR_BODY/{sign_id}_0000_0000.avi",
                f"static/videos/{nombre.lower()}.avi")
    with open(f"static/signs/{nombre.lower()}.json", "w") as f:
        json.dump({"sign": nombre, "fps": 30, "video": f"videos/{nombre.lower()}.avi",
                   "frames": frames}, f)
```

**Bloqueo actual**: no hay tabla `sign_id → nombre` para LSC50. Buscar en documentación del dataset o en `LSC50/IMU/INFO/` o ejecutar manualmente sobre los videos.

⚠️ Navegadores no reproducen `.avi` nativamente. Convertir a `.mp4`:

```bash
ffmpeg -i input.avi -c:v libx264 -crf 23 output.mp4
```

## Mejoras de retargeting

### Suavizado adicional

Aplicar **filtro de un euro** (One Euro Filter) a los landmarks crudos antes de procesarlos. Reduce jitter mejor que slerp.

### Profundidad correcta

Actualmente `Z_FACTOR = 0.65` amortigua la profundidad. Mejor: usar **landmarks 3D normalizados** de MediaPipe (`pose_world_landmarks`) en vez de los normalizados a imagen. Da escala en metros.

### Calidad de manos

Las muñecas a veces "flotan" cuando MediaPipe pierde la detección. Solución: detectar continuidad y mantener última posición conocida durante gaps cortos.

### Reglas de signo específicas LSC

Algunos signos tienen movimientos no-manuales críticos (cejas, expresión). Requiere **face landmarks** (que LSC50 ya tiene en `FACE_LANDMARKS/`) y morph targets del avatar VRM.

## Pipeline texto → señas

Actualmente: tokenizar el texto y buscar cada palabra en el índice. Limitaciones:

- No hay **glosado**: el español tiene gramática diferente al LSC.
- No hay **deletreo (fingerspelling)** cuando no existe la seña.
- No hay **animación de transición** entre señas (avatar queda en última pose y salta).

### Mejoras posibles

1. **Glosado básico**: tabla de mapeos `"buenos días" → ["bueno", "dia"]`.
2. **Fingerspelling con LSC70**: dataset de imágenes A-Z. Mostrar letras una a una cuando falta seña.
3. **Transiciones**: entre seña N y N+1, interpolar últimos K frames con primeros K frames de la siguiente.

## Calidad de avatares

Avatares actuales:

| Avatar           | Formato | Notas                                |
|------------------|---------|--------------------------------------|
| Sample VRM1      | .vrm    | Usable, modelo de pruebas            |
| Alicia           | .vrm    | VRM0, requiere `rotateVRM0`          |
| VRoid (default)  | .vrm    | Mejor calidad estética               |
| VRM.glb          | .glb    | **Sin esqueleto humanoide VRM** — no usable |
| avatar_glb       | .glb    | Igual                                |
| persona          | .glb    | Igual                                |

Para usar `.glb` se requeriría mapear manualmente nombres de huesos a VRM humanoid. No vale la pena con VRMs disponibles.

## Performance

Actualmente todo corre en CPU del cliente. Bottleneck:
- `Kalidokit.Hand.solve()` ~1-2ms por mano × 2 = 4ms.
- `vrm.update()` ~2-5ms.

Total por frame ~10ms → 100fps fácilmente. No es problema actual.

Si se añade más detalle (face landmarks, dedos del pie, etc.), considerar **calcular rotaciones offline** durante extracción y guardar quaterniones en el JSON. El frontend solo aplicaría rotaciones pre-calculadas.
