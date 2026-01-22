#!/bin/bash

echo "Starting ArC API and UI..."
echo ""

# Start API server in background
echo "[1/2] Starting FastAPI server on http://localhost:8000"
python run_api.py &
API_PID=$!

# Wait a few seconds for API to start
sleep 3

echo "[2/2] Starting Gradio UI..."
echo ""

# Trap Ctrl+C to kill both processes
trap "echo 'Stopping services...'; kill $API_PID 2>/dev/null; exit" INT TERM

# Start Gradio UI
python run_ui.py --quiet

# After Gradio starts, show the message
echo ""
echo "Both services are running:"
echo "- API: http://localhost:8000"
echo "- UI: http://localhost:7860"
echo ""
echo "Press Ctrl+C to stop both services."
echo ""

# Wait for UI process
wait

# If UI exits, kill API
kill $API_PID 2>/dev/null
