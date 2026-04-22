# Memoria Técnica y Bitácora de Desarrollo: Proyecto LSC (Abril 2026)

Este documento contiene un volcado profundo de conocimiento, decisiones arquitectónicas y "lecciones aprendidas" acumuladas durante el desarrollo del sistema de traducción LSC.

## 1. Evolución del Renderizado (Del VRM al Mannequin Engine)
Originalmente el proyecto usaba modelos `.vrm` complejos. Sin embargo, para fines de análisis lingüístico y estabilidad multiplataforma, se migró a un **Motor de Maniquí Procedural** (`translator_app.js` v5.6).
- **Decisión**: Se abandonó la cinemática inversa (IK) compleja en favor del **Posicionamiento Directo de Landmarks**.
- **Razón**: Los datos de MediaPipe Holistic suelen ser ruidosos; los solvers de IK amplificaban este ruido creando "vibraciones" antinaturales. El posicionamiento directo de cilindros entre puntos A y B resultó 100% estable.

## 2. Especificaciones de Mapeo (MediaPipe → Three.js)
El sistema utiliza una escala global y offsets específicos para centrar al avatar en una cámara de 45° de campo de visión.
- **Escala**: `1.6`
- **Offset Y**: `0.8`
- **Conversión de Ejes**:
  ```javascript
  x: (lm.x - 0.5) * SCALE
  y: -(lm.y - 0.5) * SCALE + OFFSET_Y
  z: -(lm.z || 0) * SCALE
  ```

## 3. Anatomía del Maniquí (Indices Críticos)
- **Hombros**: MediaPipe 11 (Izq), 12 (Der).
- **Cadera**: MediaPipe 23 (Izq), 24 (Der).
- **Muñecas**: MediaPipe 15 (Izq), 16 (Der). *Nota: Las mallas de manos se anclan aquí y usan un Z-nudge de +0.10 para evitar clipping con el torso.*
- **Rostro**: Se mapean 468 landmarks, pero el motor actual prioriza:
  - 13/14: Apertura de mandíbula.
  - 159/145 y 386/374: Parpadeo.
  - 66/159 y 296/386: Elevación de cejas.

## 4. Gestión de Estilos (Arquetipos)
El sistema maneja tres estados (`AVATAR_STYLE`):
- **A (Joven)**: Enfocado en claridad de señas. Usa materiales `MeshPhysicalMaterial` con clearcoat bajo.
- **C (Cyborg)**: Eliminación de rasgos humanos, ojos rectangulares emisivos. Útil para entornos de baja carga gráfica.
- **D (Mentor Senior)**: Proporciones robustas, rostro esculpido (Box-Sphere hybrid) y cabello canoso (`0xb0b0b0`).

## 5. Solución de Errores Históricos (Gotchas)
- **El Pilar Gigante**: Ocurría porque las mallas se creaban en `(0,0,0)` con altura `1` antes de recibir datos. **Solución**: `visible = false` por defecto en `buildMannequin`.
- **Fugas de Memoria**: Three.js no limpia automáticamente la GPU al remover mallas. **Solución**: Llamar a `.dispose()` en `geometry` y `material` dentro de `rebuildMannequin`.
- **Z-Fighting en Manos**: Las manos de MediaPipe a veces quedan "detrás" del pecho en la proyección 2D. **Solución**: Implementamos un `Z-nudge` progresivo hacia la cámara para codos y muñecas.

## 6. Pipeline de Backend (`lsc_api_server.py`)
- **Puerto**: 8000.
- **Flujo**:
  1. Recibe texto.
  2. `nlp_translator.py` convierte a glosas (usa diccionario local o GPT-4o fallback).
  3. `motion_synthesizer.py` busca frames de landmarks en las colecciones.
  4. Si hay múltiples glosas, aplica una interpolación lineal de 5 frames (`blending`) para suavizar el "salto" entre señas.

## 7. Próximos pasos recomendados
1. **Normalización Dinámica**: Ajustar el `SCALE` basado en la distancia entre hombros del video original para que todos los intérpretes tengan la misma altura relativa.
2. **Sistema de Ropa**: Actualmente el color de la camisa es estático por estilo. Se podría parametrizar.
3. **Optimización de Rostro**: Migrar los 468 puntos del rostro a una `BufferGeometry` única para mejorar el rendimiento en móviles.

---
*Fin del volcado de contexto. Este documento debe considerarse complementario al código fuente.*
