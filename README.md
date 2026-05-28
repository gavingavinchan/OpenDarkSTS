# DarkfleetSTS — Dark Fleet Intelligence System

AI-powered satellite imagery analysis for detecting illegal ship-to-ship (STS) transfers.

## Folder Structure
- `app.py` — Main Flask application
- `templates/` — Frontend GUI (index.html)
- `scripts/` — Utility scripts
- `notebooks/` — Training notebooks
- `docs/` — User manual
- `weights/` — Model weights (best.pt)
- `data/` — Dataset (not included)
- `AIS_Downloads/` — Downloaded AIS data

## Quick Start
1. Install dependencies: `pip install flask ultralytics opencv-python torch zstandard pandas`
2. Place best.pt in weights/ folder
3. Update BASE_DIR and BEST_WEIGHTS paths in app.py
4. Run: `python app.py`
5. Open: http://localhost:5000

## Requirements
- Python 3.10+
- NVIDIA GPU recommended
- See docs/OpenDarkSTS_User_Manual.docx for full setup guide
