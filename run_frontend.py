"""
Convenience script to start the Streamlit frontend.
Run: python run_frontend.py
"""
import os
import subprocess
import sys

if __name__ == "__main__":
    os.environ.setdefault("BACKEND_URL", "http://localhost:8000")
    print(f"\n{'='*50}")
    print(f"  ResearchPilot AI — Frontend")
    print(f"  http://localhost:8501")
    print(f"{'='*50}\n")
    subprocess.run(
        [
            sys.executable, "-m", "streamlit", "run",
            "frontend/streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--browser.gatherUsageStats", "false",
            "--server.headless", "true",
        ],
        check=True,
    )
