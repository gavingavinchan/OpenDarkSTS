# OpenDarkSTS — Dark Ship-to-Ship Transfer Detector

An open-source system for detecting candidate dark-fleet ship-to-ship (STS) transfers from Sentinel-2 satellite imagery, with automatic AIS cross-verification against the U.S. Marine Cadastre feed.

**Headline detector:** YOLO11s, held-out test mAP50 = 0.895 (0.869 single vessel, 0.920 STS pair).

---

## Features

- **YOLO-based STS detection** on Sentinel-2 tiles — two classes: `ship` (single vessel) and `ships` (STS pair).
- **AIS cross-verification** — automatic dark-status classification: `FULLY_DARK`, `PARTIALLY_DARK`, `AIS_PRESENT`, `DETECTED_ONLY`.
- **Flask GUI** with seven analyst tabs (Intelligence HQ, Scan Images, STS Events, All Detections, Live Map, Export Report, AIS Manager) and three developer tabs (Single Detect, Train Model, Reference).
- **Integrated Marine Cadastre AIS downloader** with regional presets (Gulf of Mexico, Persian Gulf, West Africa, South China Sea, Mediterranean) and custom bounding boxes.
- **CSV / JSON export** of detection records for downstream reporting.

---

## Quick start

### 1. Clone

    git clone https://github.com/gavingavinchan/OpenDarkSTS.git
    cd OpenDarkSTS

### 2. Set up a Python environment (Python 3.10+ recommended)

    python -m venv darkfleet_env
    darkfleet_env\Scripts\activate

On macOS or Linux use `source darkfleet_env/bin/activate` instead.

### 3. Install dependencies

    pip install flask ultralytics opencv-python torch matplotlib seaborn pandas folium zstandard

For NVIDIA GPUs (recommended): install PyTorch matching your CUDA version from https://pytorch.org. The app runs on CPU too but noticeably slower.

### 4. Run the app

    python app.py

Open **http://localhost:5000** in your browser.

The YOLO11s weights (`weights/best.pt`, 18 MB) ship with the repo, so no separate model download is needed.

---

## Typical workflow

1. **AIS Manager** — download the AIS daily file(s) for the region and dates you want to check.
2. **Scan Images** — upload one or more Sentinel-2 tiles, choose sensitivity (Low / Medium / High), point at the AIS folder.
3. **STS Events / All Detections** — review candidate STS pairs and every annotated tile; color-coded flash alerts highlight dark-fleet cases.
4. **Live Map** — inspect detections on an OpenStreetMap base layer.
5. **Export Report** — download CSV or JSON for downstream reporting.

Full details are in `docs/OpenDarkSTS_User_Manual.docx`.

---

## Repo layout

    OpenDarkSTS/
    app.py                    Flask backend
    templates/index.html      Single-page GUI
    weights/best.pt           Headline YOLO11s detector (18 MB)
    scripts/                  Utility scripts (bbox convert, AIS correlate, YOLO compare, ...)
    notebooks/                Training notebook
    docs/                     User manual + design notes
    data/                     Your Sentinel-2 tiles (gitignored)
    AIS_Downloads/            Downloaded AIS files (gitignored)

---

## Case study

The Gulf of Mexico case study on 23 March 2024 is documented in the accompanying SNAME Maritime Convention 2026 paper. The system correctly classifies a coordinated STS between two large tankers (EAGLE SAN PEDRO, 274 m Aframax, and SEAVOICE, 336 m Suezmax) as `AIS_PRESENT` — a positive control for the classification logic.

---

## System requirements

- Python 3.10 or newer
- NVIDIA GPU recommended (CPU inference is supported, just slower)
- ~2 GB of disk for the app + weights; more for downloaded Sentinel-2 tiles and AIS files
- Windows, macOS, or Linux (developed on Windows)

---

## License & citation

See the LICENSE file. If you use OpenDarkSTS in academic work, please cite the SNAME Maritime Convention 2026 paper.
