"""
Main application entry point for the AI Copilot for Data Teams.

This file provides multiple interfaces:
1. Web Interface (new clean version) - Default
2. Command Line Interface (CLI) - Future feature

Usage:
- For new web UI: uv run main.py --ui web (default)
- For CLI: uv run main.py --ui cli (not implemented yet)
"""

import argparse
import sys
import os
from dotenv import load_dotenv
from src.utils.logging import configure_logging

# Configure logging first
configure_logging()

# Load environment variables from .env file
load_dotenv()


def run_streamlit_app():
    """Run the new clean Streamlit application."""
    import subprocess
    
    # Get path to new app
    current_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(current_dir, "src", "web", "app.py")
    
    # Run streamlit
    cmd = [
        sys.executable, "-m", "streamlit", "run", 
        app_path,
        "--server.port", "8501",
        "--server.address", "localhost",
        "--server.headless", "false"
    ]
    
    print("🚀 Starting AI Copilot for Data Teams (New Clean Version)")
    print("📍 URL: http://localhost:8501")
    print("Press Ctrl+C to stop the server")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Streamlit: {e}")
        sys.exit(1)


def main():
    """
    Entry point for the application.
    Allows selecting between different interfaces.
    """
    parser = argparse.ArgumentParser(description="AI Copilot for Data Teams")
    parser.add_argument(
        "--ui",
        choices=["web", "cli"],
        default="web",
        help="Select user interface: web (new clean version) or cli",
    )
    
    args = parser.parse_args()
    
    if args.ui == "web":
        print("🔄 Using Streamlit interface...")
        run_streamlit_app()
    elif args.ui == "cli":
        print("❌ CLI interface not implemented yet. Use --ui web instead.")
        sys.exit(1)


if __name__ == "__main__":
    main()
