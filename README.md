# EyeCare – Eye Strain Monitoring Website

A Flask web application that reuses the original EyeCare MediaPipe Face Mesh, EAR blink detection, iris-IPD distance calibration, and weighted ESI algorithm.

## 1. Requirements

- Python 3.11 (recommended)
- A webcam
- Chrome, Edge, or Firefox
- Internet connection for Google Fonts (optional)

## 2. Create and activate a virtual environment

### Windows PowerShell
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:
```powershell
.venv\Scripts\activate.bat
```

## 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Run locally

```bash
python app.py
```

Open:
http://127.0.0.1:5000

Allow camera permission, select **Start monitoring**, sit approximately 50 cm from the camera, and select **Calibrate at 50 cm**.

## 5. GitHub and Render deployment

1. Create a GitHub repository named `eyecare`.
2. Upload all project files (do not upload `.venv` or the local database).
3. Sign in to Render and create a new **Web Service**.
4. Connect the GitHub repository.
5. Use these settings:
   - Runtime: Python
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn --workers 1 --threads 4 --timeout 120 app:app`
6. Deploy and open the HTTPS URL.
7. Allow camera permission in the browser.

`render.yaml` is included for convenience.

## 6. Important deployment notes

- Webcam access requires HTTPS on a deployed website. Render provides HTTPS.
- This first version uses one application process because MediaPipe state and session state are held in memory. Do not increase Gunicorn workers without changing the session architecture.
- SQLite is suitable for local development and an initial demonstration. On hosting platforms with ephemeral filesystems, the database may reset during redeploy/restart. Use a persistent disk or managed database before relying on long-term history.
- The server processes uploaded camera frames but does not save the frame images.
- This is a wellness indicator, not a medical diagnostic tool. ESI weights and thresholds are heuristic and should be validated before making health claims.
- A production multi-user release should use per-user session state, authentication, request limits, secure session handling, and a persistent database.

## 7. Existing algorithm retained

- MediaPipe Face Mesh with refined iris landmarks
- EAR landmark sets: left `[362, 385, 387, 263, 373, 380]`, right `[33, 160, 158, 133, 153, 144]`
- EAR threshold `0.21`; consecutive frames `2`
- Iris centers `473` and `468`
- Known IPD `6.3 cm`; calibration distance `50 cm`
- Blink baseline `17 blinks/min`
- Ideal distance `60 cm`; unsafe distance below `40 cm`
- ESI weights: blink `0.30`, closure `0.25`, distance `0.25`, screen time `0.20`
- ESI levels: below `30` low, `30–59.9` moderate, `60+` high
