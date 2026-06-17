# 01 — Arquitectura general

## Objetivo

Plataforma web que recibe **texto en español** y lo reproduce en **Lengua de Señas Colombiana (LSC)** usando un **avatar 3D humanoide** (formato VRM) animado en tiempo real.

## Flujo end-to-end

```
┌─────────────────┐     ┌─────────────────┐     ┌────────────────────┐     ┌──────────────────┐
│  Video original │ ──► │   MediaPipe     │ ──► │  Landmarks JSON    │ ──► │   Frontend web   │
│  (.mp4 / .m4v)  │     │  (Python)       │     │  (pose + manos)    │     │   (Three.js)     │
└─────────────────┘     └─────────────────┘     └────────────────────┘     └──────────────────┘
                                                                                    │
                                                                                    ▼
                                                                          ┌──────────────────┐
                                                                          │  Avatar VRM      │
                                                                          │  (huesos rotados)│
                                                                          └──────────────────┘
```

### Etapas

| # | Etapa                 | Tecnología                          | Salida                            |
|---|-----------------------|-------------------------------------|-----------------------------------|
| 1 | Captura               | Cámara → videos pre-grabados        | `LSCPROPIO/*.mp4`, `LSC50/*.avi`  |
| 2 | Extracción landmarks  | MediaPipe (Pose + Hand)             | `static/signs/<palabra>.json`     |
| 3 | Indexación            | Script Python                       | `static/signs/index.json`         |
| 4 | Servidor estático     | `python -m http.server`             | http://localhost:8765             |
| 5 | Render 3D             | Three.js + @pixiv/three-vrm         | Avatar animado en canvas WebGL    |
| 6 | Retargeting           | JavaScript (geometría vectorial)    | Rotaciones de huesos VRM          |

## Estructura del proyecto

```
prototipo/
├── extraer.py              # extrae señas de LSCS45/sample.json (formato pre-procesado)
├── extraer_videos.py       # extrae landmarks de videos LSCPROPIO usando MediaPipe
├── models/                 # modelos .task de MediaPipe
│   ├── pose_landmarker_full.task
│   └── hand_landmarker.task
├── vrm-avatar/             # avatares fuente (VRM, GLB)
├── static/                 # servido por HTTP
│   ├── index.html          # toda la lógica de retargeting + render
│   ├── avatar.vrm          # avatar por defecto
│   ├── avatars/            # alternativos (VRoid, Alicia, etc.)
│   ├── videos/             # videos originales sincronizados con landmarks
│   └── signs/
│       ├── index.json      # diccionario palabra → archivo
│       └── <palabra>.json  # landmarks por frame
└── docs/                   # esta documentación
```

## Datasets usados

| Dataset      | Tipo          | Uso actual                              |
|--------------|---------------|-----------------------------------------|
| LSCS45       | JSON pre-extraído | 5 señas demo (apoyar, ayudar, etc.) |
| LSCPROPIO    | Videos crudos | 44 palabras extraídas con MediaPipe     |
| LSC50        | Videos + CSV  | **No integrado todavía**                |
| LSC70        | Imágenes      | No aplicable (señas estáticas)          |

Ver `02-extraccion-landmarks.md` para detalle de extracción.
Ver `03-retargeting-avatar.md` para detalle del cruce landmarks → huesos.
