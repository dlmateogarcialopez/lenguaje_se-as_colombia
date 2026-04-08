@echo off
setlocal
set VENV_PYTHON=d:\LSC\venv311\Scripts\python.exe

echo ============================================================
echo   SISTEMA LSC: RECONSTRUCCION COMPLETA Y EJECUCION
echo ============================================================
echo.

echo [1/3] PASO 1: Ejecutando Pipeline de Datos (Extraccion MediaPipe)...
echo       (Esto puede tardar varios minutos debido al procesamiento de videos)
%VENV_PYTHON% d:\LSC\pipeline\run_pipeline.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] El pipeline fallo. Revisa los logs arriba.
    pause
    exit /b %ERRORLEVEL%
)
echo ✅ Pipeline completado con exito.
echo.

echo [2/3] PASO 2: Entrenando el Modelo de IA (LSTM)...
%VENV_PYTHON% d:\LSC\pipeline\train_lstm_mvp.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] El entrenamiento fallo.
    pause
    exit /b %ERRORLEVEL%
)
echo ✅ Entrenamiento completado. Modelo guardado en pipeline_output/
echo.

echo [3/3] PASO 3: Iniciando Servidor LSC Unificado...
echo       Abre tu navegador en: http://localhost:8000
echo.
%VENV_PYTHON% d:\LSC\pipeline\lsc_api_server.py

pause
