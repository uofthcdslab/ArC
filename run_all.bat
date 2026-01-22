@echo off
echo Starting ArC API and UI...
echo.

REM Start API server in background
echo [1/2] Starting FastAPI server on http://localhost:8000
start "ArC API Server" cmd /k "python run_api.py"

REM Wait a few seconds for API to start
timeout /t 3 /nobreak >nul

echo [2/2] Starting Gradio UI...
echo.

REM Start Gradio UI and wait for it to be ready
python run_ui.py --quiet

REM This will only show after Gradio starts
echo.
echo Both services are running:
echo - API: http://localhost:8000
echo - UI: http://localhost:7860
echo.
echo Press Ctrl+C to stop the UI. Close the API window separately.
echo.

