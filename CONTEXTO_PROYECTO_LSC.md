# Contexto Maestro: Proyecto Traductor LSC 3D

Este documento proporciona el contexto completo, arquitectónico y operativo del proyecto de traducción de Lengua de Señas Colombiana (LSC) para futuros desarrolladores o modelos de IA.

## 1. Misión del Proyecto
Desarrollar un sistema de traducción bidireccional (actualmente enfocado de Español a LSC) que utilice un avatar humanoide 3D para interpretar texto en tiempo real, manteniendo la fidelidad lingüística y la naturalidad del movimiento.

## 2. Tecnologías Core
### Frontend (Renderizado e Interfaz)
- **Three.js**: Motor principal de renderizado 3D.
- **Vanilla JavaScript**: Lógica de aplicación y gestión de landmarks.
- **HTML5/CSS3 (Glassmorphism)**: Interfaz de usuario premium y responsiva.
- **Kalidokit**: (Opcional/Referencia) para cinemática inversa, aunque el motor v5.6 actual usa posicionamiento directo de landmarks para estabilidad.

### Backend (API y Lógica)
- **FastAPI (Python 3.11)**: Servidor de alto rendimiento.
- **FFmpeg**: Conversión de video de referencia en tiempo real.
- **MediaPipe**: Extracción de landmarks de pose, manos y rostro.
- **NumPy / Pandas**: Procesamiento de datos masivos de landmarks.

## 3. Arquitectura del Sistema

### A. Pipeline de Datos (`/pipeline`)
El sistema procesa videos reales para extraer "colecciones maestras" de landmarks.
- **Extractor**: Convierte video a `.csv` / `.pkl` con 1629 coordenadas por frame.
- **Imputación**: Rellena datos ruidos de landmarks usando promedios temporales.

### B. Motor de Síntesis (`motion_synthesizer.py`)
- Recibe glosas (ej. `HOLA`, `AMIGO`) y busca en la base de datos de landmarks.
- Aplica **Blending Temporal**: Transiciones suaves entre el final de una seña y el inicio de la siguiente.
- **Proceduralismo**: Si una seña no existe, el sistema puede ensamblar componentes o deletrear (fingerspelling).

### C. Motor de Maniquí v5.6 (`webapp/translator_app.js`)
Es un motor de maniquí humanoide construido desde cero con primitivas de Three.js.
- **Arquetipos**:
  - **Estilo A (Joven Profesional)**: Mallas humanas estilizadas.
  - **Estilo C (Cyborg)**: Look futurista minimalista.
  - **Estilo D (Mentor Senior)**: Proporciones de persona mayor con texturas específicas.
- **Estabilización**: Usa un sistema de "ocultamiento en el arranque" para evitar artefactos geométricos (pilares) antes de recibir datos.

## 4. Estado Actual y Logros (Abril 2026)
- **Estabilización del Render**: Se resolvieron los problemas de "pilares azules" y "bloques masivos" mediante una gestión estricta del ciclo de vida de la GPU y la eliminación de mallas redundantes (cuello/relleno de torso).
- **Consistencia Lingüística**: El sistema traduce correctamente frases simples y términos del diccionario LSC50.
- **UI/UX**: Dashboard intuitivo con panel de video de referencia y selector de arquetipos.

## 5. Skills Disponibles (Agentes)
El proyecto cuenta con "Skills" especializadas que deben ser consultadas para tareas específicas:
- `data_engineer_lsc_pipeline`: Experto en ingestión y MediaPipe.
- `ai_ml_engineer_lsc`: Para modelos Transformer/LSTM de recognition.
- `frontend_developer_vrm_lsc`: Para WebGL y Three.js avanzado.
- `lsc_nlp_translator_expert`: Para lingüística y gramática LSC.

## 6. Instrucciones de Ejecución
1. **Servidor**: Ejecutar `python pipeline/lsc_api_server.py`. (Puerto 8000).
2. **Cliente**: Abrir `webapp/translator.html` en el navegador. (Usa cache-busting v4).

## 7. Desafíos Pendientes
- **Ampliación de Vocabulario**: Ingestar más videos de la carpeta `LSCPROPIO`.
- **Reconocimiento (Sign-to-Text)**: El módulo de captura de cámara aún es experimental.
- **Expresiones Faciales**: Refinar la intensidad de mímica facial basada en landmarks MediaPipe.

---
*Este documento es la fuente de verdad para la transferencia de contexto del proyecto LSC.*
