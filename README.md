# ID Portrait Extractor (Streamlit)

A minimal, no‑training solution to extract portrait photo(s) from ID card images. It uses MediaPipe for fast face detection, crops detected faces, and lets you download the main portrait or all faces as a ZIP.

## Project Goals
- Provide a simple baseline that “just works” on many IDs without training.
- Offer a clean structure students can extend (deskewing, custom detectors, etc.).

## Project Structure
```
.
├─ src/
│  ├─ app.py                         # Streamlit UI entrypoint
│  └─ face_extractor/
│     ├─ __init__.py                 # Exports helper functions
│     └─ detector.py                 # MediaPipe detection + cropping
├─ requirements.txt                  # Unpinned latest versions
├─ .gitignore                        # Ignore venv, caches, outputs
└─ README.md
```

## Prerequisites
- Python 3.8–3.12 (tested with 3.8 on Windows)
- Internet access for the first `pip install`

## Setup (Students)
1) Create and activate a virtual environment

- Windows (PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

- macOS/Linux (bash or zsh)
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) Install dependencies (latest versions)
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If your network is slow, re‑run the install; use a higher timeout if needed:
```bash
pip install --default-timeout 180 -r requirements.txt
```

3) Run the app
```bash
streamlit run src/app.py
```

Open the local URL shown in the terminal.

### Optional: Install dependencies at runtime
This repo also includes a small bootstrapper that can install missing packages when the app starts. This is intended for local development or environments that allow on‑the‑fly installs and may add startup time.

Enable it by setting an environment variable before launching Streamlit:

- Windows PowerShell
```powershell
$env:ALLOW_RUNTIME_INSTALL = "1"
streamlit run src/app.py
```

- macOS/Linux
```bash
ALLOW_RUNTIME_INSTALL=1 streamlit run src/app.py
```

Notes for Streamlit Community Cloud
- This project uses OpenCV’s Haar cascade by default (no external system libs). The dependency is `opencv-python-headless`, which is Cloud‑friendly.
- MediaPipe is optional. If you want to enable it, add `mediapipe` to `requirements.txt` (and ensure the selected Python version is supported by the wheel). The app automatically uses MediaPipe when available.
- If deploy builds fail due to Python version compatibility, add `runtime.txt` with a supported version (e.g., `3.10`).

## Using the App
- Upload an image of an ID card (`.jpg`, `.jpeg`, `.png`, `.webp`).
- Sidebar options:
  - Min confidence: detector threshold
  - Crop margin (%): extra pixels around the face box
  - Return: largest only vs all faces
  - Max faces: limit when “All faces” is selected
- View: detection overlay and cropped portraits
- Downloads:
  - “Download main portrait” → `portrait_main.jpg`
  - “Download all as ZIP” → `portraits.zip`

## Troubleshooting
- ImportError: attempted relative import with no known parent package
  - Fixed in `src/app.py`. If needed, run with `PYTHONPATH=src` (macOS/Linux) or `$env:PYTHONPATH="src"` (Windows) before `streamlit run src/app.py`.
- Slow/failed `pip install`
  - Re‑run with a higher timeout as shown above. Ensure your Python version is supported.
- No faces detected
  - Lower the confidence slider and try a clearer, frontal image. Consider snapping a tighter photo of the ID to avoid background faces.

## Extending (Assignments/Ideas)
- Deskew and crop the ID card via contour + perspective warp before face detection.
- Train a lightweight detector (YOLO) to find the portrait window specifically.
- Add EXIF stripping and temporary storage controls for privacy.
- Batch mode: accept multiple images and export a results CSV.

## Notes
- MediaPipe is robust for frontal faces, but may pick up background faces if the photo includes people beyond the card. Deskewing or a trained detector can mitigate this.
- The app runs offline after initial installation.
