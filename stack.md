# Arquitectura Técnica - Proyecto LSC (Lengua de Señas Colombiana)

El proyecto LSC es un motor integral diseñado para capturar, procesar, ingestar y sintetizar secuencias tridimensionales de Lengua de Señas Colombiana. La arquitectura se divide en tres bloques principales: el **Pipeline de Extracción (Data Engineering)**, el **Motor API (NLP & Synth)**, y la **Aplicación Frontend (WebGL Analytics)**.

---

## 1. Pipeline de Extracción y Procesamiento de Datos (Python/ML)
Este bloque se encarga de convertir videos de distintos orígenes a secuencias numéricas (Landmarks) pre-calculadas y normalizadas, preparadas para ML o síntesis rápida.

*   **Tecnologías:** `Python 3.11`, `OpenCV`, `MediaPipe Holistic`, `Pandas`, `NumPy`.
*   **Archivos Clave:**
    *   `pipeline/ingest_lscpropio.py`: Ingiere videos propios del equipo (`LSCPROPIO/`) extraídos frame por frame utilizando MediaPipe Holistic.
    *   `pipeline/convert_lsc70.py` & `pipeline/unify_impute.py`: Fusionan e imputan datos ruidosos desde múltiples fuentes (LSC50, LSC70) a un marco padronizado intermedio (1629 coordenadas espaciales: Pose 99 + Mano Izq 63 + Mano Der 63 + Rostro 1404).
*   **Resultados:** Genera grandes repositorios intermedios `.csv` y `.pkl` (`pipeline_output/lsc_dataset_final.csv`) que alimentan el traductor en tiempo real sin requerir reconstrucción de video al vuelo.

---

## 2. LSC Motor Backend (API y Traducción)
Servidor de arquitectura RESTful que recibe texto en español, lo traduce lingüísticamente a glosas LSC y luego compone la secuencia tridimensional de movimiento final.

*   **Tecnologías:** `FastAPI`, `Uvicorn`.
*   **Archivos Clave:**
    *   `pipeline/lsc_api_server.py`: Punto de entrada de la API. Expone puertos HTTP (8000) e interactúa de intermediario.
    *   `pipeline/lsc_nlp_translator_expert`: Módulos de Procesamiento de Lenguaje Natural para aislar reglas gramaticales de Español → Español estructurado LSC → Glosas Exactas.
    *   `pipeline/motion_synthesizer.py`: El corazón procedural. Recibe secuencias de señas puras y compone marcos contiguos (blending de animaciones) cargando los `.csv` dinámicamente. Contiene lógica procedural manual que esquiva las dependencias de modelos rotos (ej. síntesis multi-mano usando posiciones relativas).
*   **Salida:** Un payload JSON estructurado bajo esquema VRM-Skeleton que la Web toma para mover el avatar.

---

## 3. Visualización Frontend (WebGL y 3D)
Aplicación de navegador que interpreta las ecuaciones JSON enviadas por el API Server y las renderiza en un Avatar humanoide o un Esqueleto de Análisis.

*   **Tecnologías:** `HTML5/CSS3`, `Vanilla JavaScript`, `Three.js` (Motor 3D), `Kalidokit` (Inverse Kinematics solver), estandar `VRM` para mallas corporales de anime/humanos.
*   **Archivos Clave:**
    *   `webapp/translator.html` / `visualizer.html`: Las pantallas del dashboard primario del analista.
    *   `webapp/translator_app.js`: Script principal de la Web que solicita el modelo 3D VRM (`models/Ashtra.vrm`).
    *   **Rigging Dinámico**: Implementa Solvers de Kalidokit que transcriben las coordenadas espaciales vectoriales de LSC a rotaciones de articulaciones (Euler/Quaternions) entendibles por Three.js en la malla ósea.

---

## Flujo Operativo Completo (User Flow)
1. **Usuario**: Ingresa un texto: *"Quiero caminar en la mañana"*.
2. **FastAPI**: Llama al NLP local y rompe las oraciones → `['QUERER', 'CAMINAR', 'MAÑANA']`
3. **Motion Synthesizer**: Busca `QUERER`, `CAMINAR` en base de datos.
     * Si las encuentra: extrae las coordenadas 3D de las colecciones maestras CSV (LSC50, LSCPROPIO).
     * Si falta: genera señas procedurales basadas en ensamblaje temporal (espalda + rotación).
4. **Respuesta**: El backend despliega JSON a la página Web (puerto 8000).
5. **ThreeJS**: Aplica transiciones suaves (blending) en el navegador mientras reproduce el avatar iterando a 30 FPS.
