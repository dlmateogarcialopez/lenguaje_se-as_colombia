# Documentación técnica — Traductor LSC

Esta carpeta documenta cómo el prototipo convierte landmarks de MediaPipe en animación de un avatar 3D VRM.

## Lectura recomendada

1. **[01-arquitectura-general.md](01-arquitectura-general.md)** — visión global, flujo de datos, estructura del proyecto.
2. **[02-extraccion-landmarks.md](02-extraccion-landmarks.md)** — cómo se procesan los videos con MediaPipe y qué se guarda.
3. **[03-retargeting-avatar.md](03-retargeting-avatar.md)** — **núcleo técnico**: cómo se traducen landmarks 2D/3D a rotaciones de huesos VRM.
4. **[04-pendientes-y-mejoras.md](04-pendientes-y-mejoras.md)** — integración de LSC50, mejoras de calidad, glosado.

## TL;DR del cruce

```
Video                                                                    Avatar 3D
─────                                                                    ─────────
[frame N]                                                                [hueso leftUpperArm]
   │                                                                        ▲
   ▼                                                                        │ slerp 55%
MediaPipe Pose:                                                             │
  landmark[11] = hombro izq (x,y,z)              ┌─────────────────────┐    │
  landmark[13] = codo izq                        │  setFromUnitVectors │ ───┘
  landmark[15] = muñeca izq                      │  (restDir, target)  │
   │                                             └─────────────────────┘
   ▼                                                        ▲
dirección = normalize(landmark[13] - landmark[11])          │
   │                                                        │
   │  ajustar ejes (Y invertida, Z amortiguada)             │
   │                                                        │
   ▼                                                        │
targetDir en espacio mundo  ──────► espacio local del padre ┘
                              (multiplicar por inversa
                               del quaternion del hombro padre)
```

Repetir para cada hueso del esqueleto. Manos se procesan aparte con `Kalidokit.Hand.solve()` para los curls de los dedos. Asignación de manos por cercanía a las muñecas del cuerpo (la etiqueta `Left/Right` de MediaPipe no es confiable).

## Archivos clave del código

| Archivo                                | Función                                              |
|----------------------------------------|------------------------------------------------------|
| `prototipo/extraer.py`                 | Extrae 5 señas de `LSCS45/sample.json`              |
| `prototipo/extraer_videos.py`          | Extrae 44 señas de videos `LSCPROPIO/*` con MediaPipe |
| `prototipo/static/index.html`          | **Todo el frontend**: render, retargeting, UI       |
| `prototipo/static/signs/index.json`    | Diccionario palabra → archivo de seña                |
| `prototipo/static/signs/<palabra>.json`| Landmarks por frame de una seña                      |

## Cómo correr

```bash
cd prototipo/static
python -m http.server 8765
```

Abrir http://localhost:8765
