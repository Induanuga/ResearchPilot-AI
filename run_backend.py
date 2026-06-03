"""
Convenience script to start the FastAPI backend.
Run: python run_backend.py
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import uvicorn
from backend.config import get_settings

if __name__ == "__main__":
    settings = get_settings()
    print(f"\n{'='*50}")
    print(f"  ResearchPilot AI — Backend")
    print(f"  http://localhost:{settings.backend_port}/api/docs")
    print(f"{'='*50}\n")
    uvicorn.run(
        "backend.main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
