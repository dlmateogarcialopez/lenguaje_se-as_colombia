# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a **data-only repository** (not a git repo, no build/test/lint commands) containing four Colombian Sign Language (Lengua de Señas Colombiana, LSC) datasets. Work here typically means analyzing, processing, or extracting features from these datasets with Python (the reference code in `LSCS45/README.ipynb` uses `mediapipe`, `opencv-python`, and `numpy`).

## Datasets

### LSC50/ — multimodal recordings of 50 dynamic signs
File naming convention everywhere: `SSSS_PPPP_RRRR` = sign ID (0000–0049) _ signer ID (0000–0004) _ repetition (0000–0003) → 1000 samples per modality.

- `VIDEOS/` — four synchronized modalities, 1000 `.avi` each: `COLOR_BODY`, `COLOR_FACE`, `GRAY_DEPTH`, `GRAY_IR`
- `LANDMARKS/` — MediaPipe landmarks as CSV (one row per frame, columns `landmark_N_x/y/z`, normalized coordinates): `BODY_LANDMARKS` (33 pose points), `FACE_LANDMARKS`, `HANDS_LANDMARKS/LEFT_HAND_LANDMARKS` and `RIGHT_HAND_LANDMARKS`
- `IMU/` — inertial capture per volunteer: `RAW/` (.xlsx), `STO/` (OpenSim quaternion .sto, DataRate=120), `OUT_OPENSIM/` (.mot inverse-kinematics output), `INFO/` (anthropometric measurements per volunteer .txt, sampling frequency 125 Hz, and `OpenSim_Model.osim`)
- `WEB_BODY/` — 5 `.webm` videos

### LSC70/ — static sign images (fingerspelling alphabet + numbers)
Inside `LSC70/LSC70/` (plus the original archive `LSC70.zip`, 945 MB):
- `LSC70AN/PerXX/<SIGN>/` — 70 persons (Per01–Per70), signs A–Z, NN, digits, MIL, MILLON; 6 images per sign named `PerXX_<SIGN>_N.jpg`
- `LSC70ANH/` — same structure (hand-region variant)
- `LSC70W/PerXX/<WORD>/` — 10 word signs (HOLA, BUENAS, DIAS, TARDES, NOCHES, NOMBRE, YO, GUSTAR, ANNOS, LICOR)

### LSCPROPIO/ — custom word videos
45 word folders (days, months, time words, technology vocabulary) with videos named `<word>-personaN.mp4` / `.m4v`. Some `*-persona2.mp4` files sit loose in the root rather than in folders.

### LSCS45/ — landmark dataset of 45 signs in 3 categories
Categories: Cordialidad, Colores, Números (full sign list in `README.ipynb`, bilingual ES/EN).
- `datos.json` — **52 GB**. Never load with `json.load`; use a streaming parser (e.g. `ijson`) or work with `sample.json`. Structure: `Signer_N > Category > SignName > vid_N > rep_N > frame_N > {pose, l_hand, r_hand} > {x: [...], y: [...]}` (normalized MediaPipe coordinates)
- `sample.json` — small sample with the same schema
- `README.ipynb` — reference code for visualizing landmarks and re-extracting them with MediaPipe (models referenced from https://github.com/juanesmz/SignCapture.git)

## Practical notes

- The repo holds ~35k images and 4k videos/CSVs; avoid unbounded recursive listings (`find` without `-maxdepth`) — output gets huge.
- Folder/file names use Spanish words and may contain spaces and non-ASCII characters (`año`, `a veces`) — always quote paths.
