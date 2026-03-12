@echo off
echo ==========================================
echo  LSC - Sistema de Traduccion Texto a Senas
echo ==========================================
echo.

echo [1/1] Iniciando Servidor LSC Unificado (Puerto 8000)...
start "LSC Unified Server" cmd /c "d:\LSC\venv311\Scripts\python.exe d:\LSC\pipeline\lsc_api_server.py"

echo.
echo Esperando que el servidor inicie...
timeout /t 5 /nobreak > nul

echo.
echo ==========================================
echo  Sistema LSC Listo!
echo ==========================================
echo.
echo  Abre tu navegador en:
echo  http://localhost:8000
echo.
pause
