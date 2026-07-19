@echo off
rem Kilnform launcher: AI backend + web UI + your default browser
cd /d "%~dp0"

echo [Kilnform] Starting the AI backend (127.0.0.1:8000)...
start "Kilnform AI backend" cmd /k "cd /d "%~dp0backend" && .venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000"

echo [Kilnform] Starting the web UI (127.0.0.1:5173)...
start "Kilnform Web" cmd /k "cd /d "%~dp0" && npm run dev"

timeout /t 5 /nobreak >nul
start http://127.0.0.1:5173
echo [Kilnform] Ready. Your browser should now be on http://127.0.0.1:5173
