"""
Run script for the completely rewritten AI Copilot web application.
"""
import subprocess
import sys
import os

def main():
    """Run the new Streamlit application."""
    # Get the directory containing this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(current_dir, "app_new.py")
    
    # Run streamlit
    cmd = [
        sys.executable, "-m", "streamlit", "run", 
        app_path,
        "--server.port", "8501",
        "--server.address", "localhost",
        "--server.headless", "false"
    ]
    
    print("🚀 Starting AI Copilot for Data Teams (New Version)")
    print("📍 URL: http://localhost:8501")
    print("Press Ctrl+C to stop the server")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()