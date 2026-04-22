"""
ingest_lscpropio.py — Video Ingestor para LSCPROPIO

Procesa videos (.mp4, .m4v) de señas LSC propias, extrayendo landmarks
corporales con MediaPipe Holistic (Pose 33 + Hands 21×2 + Face 468).

Genera un CSV intermedio (`lscpropio_interim.csv`) con el mismo esquema
estandarizado que los demás ingestores del pipeline.

Autor: LSC Pipeline Team
"""

import os
import re
import glob
import logging
from typing import Optional, List, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import mediapipe.solutions.holistic as mp_holistic
    import mediapipe.solutions.drawing_utils as mp_drawing
    import mediapipe.solutions.drawing_styles as mp_drawing_styles
except ImportError:
    import mediapipe.python.solutions.holistic as mp_holistic
    import mediapipe.python.solutions.drawing_utils as mp_drawing
    import mediapipe.python.solutions.drawing_styles as mp_drawing_styles

import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------------------------------------------------------
# Constantes de landmarks MediaPipe Holistic
# ---------------------------------------------------------------------------
NUM_POSE_LANDMARKS = 33
NUM_HAND_LANDMARKS = 21
NUM_FACE_LANDMARKS = 468

VIDEO_EXTENSIONS = ('.mp4', '.m4v', '.avi', '.mov', '.mkv', '.webm')

# Regex para parsear nombre de archivo: <seña>-persona<N>.ext  o  <seña>persona<N>.ext
_FILENAME_PATTERN = re.compile(
    r'^(?P<sign>.+?)[_\-]?persona(?P<signer>\d+)',
    re.IGNORECASE,
)


class LSCPropioIngestor:
    """Ingesta videos de LSCPROPIO y extrae landmarks con MediaPipe Holistic.

    Attributes:
        base_dir: Ruta raíz del directorio LSCPROPIO.
        max_videos: Límite opcional de videos a procesar (None = todos).
        frame_step: Cada cuántos frames se procesa (1 = todos, 2 = cada 2, …).
        chunk_size: Cada cuántos registros se hace flush a CSV (controla RAM).
    """

    def __init__(
        self,
        base_dir: str,
        max_videos: Optional[int] = None,
        frame_step: int = 1,
        chunk_size: int = 50_000,
    ) -> None:
        self.base_dir = base_dir
        self.max_videos = max_videos
        self.frame_step = max(1, frame_step)
        self.chunk_size = chunk_size
        self._visual_tested = False

    # ------------------------------------------------------------------
    # Descubrimiento de archivos
    # ------------------------------------------------------------------
    def discover_videos(self) -> List[Tuple[str, str, str]]:
        """Recorre LSCPROPIO/ y devuelve una lista de (ruta, sign_label, signer_id).

        Lógica de etiquetado:
          - Si el video está dentro de una subcarpeta → sign_label = nombre de carpeta.
          - Si está suelto en raíz → sign_label se extrae del prefijo del nombre.
          - signer_id se extrae del sufijo `personaN` en el nombre del archivo.
        """
        videos: List[Tuple[str, str, str]] = []

        for root, _dirs, files in os.walk(self.base_dir):
            for fname in sorted(files):
                if not fname.lower().endswith(VIDEO_EXTENSIONS):
                    continue
                filepath = os.path.join(root, fname)

                # Determinar sign_label
                parent = os.path.basename(root)
                if parent.upper() == 'LSCPROPIO':
                    # Archivo suelto en raíz → extraer del nombre
                    sign_label = self._parse_sign_from_filename(fname)
                else:
                    # Dentro de una subcarpeta → la carpeta ES la seña
                    sign_label = parent.strip().lower()

                # Determinar signer_id
                signer_id = self._parse_signer_from_filename(fname)

                videos.append((filepath, sign_label, signer_id))

        logging.info(
            f"Descubiertos {len(videos)} videos en {self.base_dir} "
            f"({len(set(v[1] for v in videos))} etiquetas únicas)"
        )
        return videos

    # ------------------------------------------------------------------
    # Parsing de metadatos desde el nombre de archivo
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_sign_from_filename(filename: str) -> str:
        """Extrae la etiqueta de seña desde el nombre del archivo.

        Ejemplo: 'energia-electrica-persona2.mp4' → 'energia-electrica'
                 'computadorpersona2.mp4'          → 'computador'
        """
        stem = os.path.splitext(filename)[0]
        match = _FILENAME_PATTERN.match(stem)
        if match:
            return match.group('sign').strip('-_ ').lower()
        # Fallback: usar el nombre completo sin extensión
        return stem.strip().lower()

    @staticmethod
    def _parse_signer_from_filename(filename: str) -> str:
        """Extrae el ID del señante desde el nombre del archivo.

        Ejemplo: 'enero-persona1.mp4' → 'persona1'
        """
        stem = os.path.splitext(filename)[0]
        match = _FILENAME_PATTERN.match(stem)
        if match:
            return f"persona{match.group('signer')}"
        return 'unknown'

    # ------------------------------------------------------------------
    # Extracción de landmarks de un solo video
    # ------------------------------------------------------------------
    def _extract_video_landmarks(
        self,
        video_path: str,
        sign_label: str,
        signer_id: str,
    ) -> List[dict]:
        """Procesa un video y extrae landmarks frame a frame.

        Returns:
            Lista de diccionarios, uno por frame procesado.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.warning(f"No se pudo abrir el video: {video_path}")
            return []

        video_id = os.path.splitext(os.path.basename(video_path))[0]
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        logging.debug(
            f"  Procesando '{video_id}': {total_frames} frames @ {fps:.1f} FPS"
        )

        records: List[dict] = []

        with mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as holistic:
            frame_idx = 0
            processed_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Muestreo: procesar solo cada N frames
                if frame_idx % self.frame_step != 0:
                    frame_idx += 1
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(frame_rgb)

                frame_data = self._results_to_dict(
                    results, sign_label, signer_id, video_id, processed_idx
                )
                records.append(frame_data)

                # Validación visual (una sola vez en toda la ejecución)
                if not self._visual_tested:
                    self._save_visual_test(frame, results, video_id)
                    self._visual_tested = True

                processed_idx += 1
                frame_idx += 1

        cap.release()
        return records

    # ------------------------------------------------------------------
    # Conversión de resultados MediaPipe a diccionario plano
    # ------------------------------------------------------------------
    @staticmethod
    def _results_to_dict(
        results,
        sign_label: str,
        signer_id: str,
        video_id: str,
        frame_id: int,
    ) -> dict:
        """Convierte los resultados de MediaPipe Holistic a un dict plano.

        Esquema de columnas idéntico a convert_lsc70.py:
          pose_i_x/y/z  (i=0..32)
          l_hand_i_x/y/z (i=0..20)
          r_hand_i_x/y/z (i=0..20)
          face_i_x/y/z   (i=0..467)
        """
        data: dict = {
            'source': 'LSCPROPIO',
            'video_id': video_id,
            'frame_id': frame_id,
            'sign_label': sign_label,
            'signer': signer_id,
        }

        # --- Pose (33 landmarks) ---
        if results.pose_landmarks:
            for i, lm in enumerate(results.pose_landmarks.landmark):
                data[f'pose_{i}_x'] = lm.x
                data[f'pose_{i}_y'] = lm.y
                data[f'pose_{i}_z'] = lm.z
        else:
            for i in range(NUM_POSE_LANDMARKS):
                data[f'pose_{i}_x'] = np.nan
                data[f'pose_{i}_y'] = np.nan
                data[f'pose_{i}_z'] = np.nan

        # --- Left Hand (21 landmarks) ---
        if results.left_hand_landmarks:
            for i, lm in enumerate(results.left_hand_landmarks.landmark):
                data[f'l_hand_{i}_x'] = lm.x
                data[f'l_hand_{i}_y'] = lm.y
                data[f'l_hand_{i}_z'] = lm.z
        else:
            for i in range(NUM_HAND_LANDMARKS):
                data[f'l_hand_{i}_x'] = np.nan
                data[f'l_hand_{i}_y'] = np.nan
                data[f'l_hand_{i}_z'] = np.nan

        # --- Right Hand (21 landmarks) ---
        if results.right_hand_landmarks:
            for i, lm in enumerate(results.right_hand_landmarks.landmark):
                data[f'r_hand_{i}_x'] = lm.x
                data[f'r_hand_{i}_y'] = lm.y
                data[f'r_hand_{i}_z'] = lm.z
        else:
            for i in range(NUM_HAND_LANDMARKS):
                data[f'r_hand_{i}_x'] = np.nan
                data[f'r_hand_{i}_y'] = np.nan
                data[f'r_hand_{i}_z'] = np.nan

        # --- Face (468 landmarks) ---
        if results.face_landmarks:
            for i, lm in enumerate(results.face_landmarks.landmark):
                data[f'face_{i}_x'] = lm.x
                data[f'face_{i}_y'] = lm.y
                data[f'face_{i}_z'] = lm.z
        else:
            for i in range(NUM_FACE_LANDMARKS):
                data[f'face_{i}_x'] = np.nan
                data[f'face_{i}_y'] = np.nan
                data[f'face_{i}_z'] = np.nan

        return data

    # ------------------------------------------------------------------
    # Validación visual
    # ------------------------------------------------------------------
    def _save_visual_test(self, img: np.ndarray, results, video_id: str) -> None:
        """Guarda una imagen con los landmarks dibujados para validación visual."""
        annotated = img.copy()
        try:
            mp_drawing.draw_landmarks(
                annotated,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
            )
            mp_drawing.draw_landmarks(
                annotated,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
            )
            mp_drawing.draw_landmarks(
                annotated,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
            )
            mp_drawing.draw_landmarks(
                annotated,
                results.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS,
            )
        except Exception as e:
            logging.warning(f"Error dibujando landmarks de validación: {e}")

        out_path = os.path.join(config.OUTPUT_DIR, f"lscpropio_validation_{video_id}.jpg")
        cv2.imwrite(out_path, annotated)
        logging.info(f"Imagen de validación visual guardada en {out_path}")

    # ------------------------------------------------------------------
    # Proceso principal
    # ------------------------------------------------------------------
    def process(self) -> pd.DataFrame:
        """Ejecuta la ingesta completa de LSCPROPIO.

        Descubre videos, extrae landmarks, y persiste un CSV intermedio
        por chunks para controlar el uso de memoria.

        Returns:
            DataFrame con todos los frames procesados (puede ser vacío
            si se usó persistencia incremental en chunks grandes).
        """
        logging.info("=" * 60)
        logging.info("  INICIO INGESTA LSCPROPIO")
        logging.info("=" * 60)

        videos = self.discover_videos()
        if not videos:
            logging.warning("No se encontraron videos en LSCPROPIO.")
            return pd.DataFrame()

        if self.max_videos is not None:
            videos = videos[: self.max_videos]
            logging.info(f"Limitado a {self.max_videos} videos para esta ejecución.")

        out_path = os.path.join(config.OUTPUT_DIR, "lscpropio_interim.csv")
        if os.path.exists(out_path):
            os.remove(out_path)

        all_records: List[dict] = []
        first_chunk = True
        total_frames_written = 0
        successful_videos = 0
        failed_videos = 0
        label_counts: dict = {}

        for video_path, sign_label, signer_id in tqdm(videos, desc="Ingesta LSCPROPIO"):
            try:
                records = self._extract_video_landmarks(video_path, sign_label, signer_id)
                if records:
                    all_records.extend(records)
                    successful_videos += 1
                    label_counts[sign_label] = label_counts.get(sign_label, 0) + len(records)
                else:
                    failed_videos += 1
                    logging.warning(f"Sin frames extraídos de: {video_path}")
            except Exception as e:
                failed_videos += 1
                logging.error(f"Error procesando {video_path}: {e}")

            # Flush a disco si acumulamos suficientes registros
            if len(all_records) >= self.chunk_size:
                df_chunk = pd.DataFrame(all_records)
                df_chunk.to_csv(out_path, mode='a', index=False, header=first_chunk)
                total_frames_written += len(all_records)
                first_chunk = False
                all_records = []
                logging.info(f"Chunk escrito al disco — {total_frames_written} frames acumulados.")

        # Escribir registros restantes
        if all_records:
            df_chunk = pd.DataFrame(all_records)
            df_chunk.to_csv(out_path, mode='a', index=False, header=first_chunk)
            total_frames_written += len(all_records)

        # Resumen final
        logging.info("=" * 60)
        logging.info("  RESUMEN INGESTA LSCPROPIO")
        logging.info(f"  Videos procesados OK : {successful_videos}")
        logging.info(f"  Videos fallidos      : {failed_videos}")
        logging.info(f"  Total frames escritos: {total_frames_written}")
        logging.info(f"  Etiquetas únicas     : {len(label_counts)}")
        logging.info(f"  CSV guardado en      : {out_path}")
        logging.info("=" * 60)

        # Devolver DataFrame del último chunk (o todo si cabe en memoria)
        if os.path.exists(out_path):
            return pd.read_csv(out_path)
        return pd.DataFrame()


if __name__ == "__main__":
    ingestor = LSCPropioIngestor(
        config.LSCPROPIO_DIR,
        max_videos=None,  # Process all videos
        frame_step=2,  # Procesar cada 2 frames
    )
    df = ingestor.process()

    if not df.empty:
        logging.info(f"Shape final: {df.shape}")
        logging.info(f"Columnas (primeras 20): {df.columns.tolist()[:20]}")
        logging.info(f"Etiquetas: {df['sign_label'].unique().tolist()}")
