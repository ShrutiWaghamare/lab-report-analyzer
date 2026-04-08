"""
START THE SERVER
================
Just run this from your project root:

    python run.py

Then open:  http://localhost:8000
"""

import os
import sys
from dotenv import load_dotenv
load_dotenv()


# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  Lab Report Analyzer")
    print("="*50)
    print("  Open in browser → http://localhost:8000")
    print("  API docs        → http://localhost:8000/docs")
    print("  Stop server     → Ctrl ++ C")
    print("="*50 + "\n")

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["api", "agents", "core", "ocr"],
    )
