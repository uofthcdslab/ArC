"""Run the Gradio UI"""
import sys
from ui.gradio_app import app

if __name__ == "__main__":
    # Check if running standalone or from run_all script
    if len(sys.argv) > 1 and sys.argv[1] == "--quiet":
        # Quiet mode - minimal output
        pass
    else:
        print("Starting ArC Gradio UI...")
        print("Make sure the API server is running: python run_api.py")
        print("UI will be available at: http://localhost:7860")
    
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )
