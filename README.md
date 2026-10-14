# IntelliWatch

Privacy-aware physical asset monitoring for finance branches — local Streamlit MVP.

This prototype helps a reviewer compare a **baseline camera view** of an illustrative office asset (printer, monitor, laptop, keyboard) with a **new photo of the same view**. It uses a pretrained **YOLOv8n** model for object *categories* and simple OpenCV image comparison. It does **not** identify a unique physical device, does **not** perform face recognition, and does **not** confirm theft.

Categories are **examples**, not confirmed L&T Finance inventory.

## What this MVP does

- Dashboard of registered assets, latest “detected” counts, open warnings, and recent events
- Asset registration with a locally stored baseline image
- Image monitoring: YOLO boxes + baseline/current comparison + a review status
- Event history with search/filter and local evidence images
- Labelled **demo records** so the UI can be shown without real photos

## Requirements

- Windows 10/11
- Python 3.11 or 3.12 recommended (this project was set up against **Python 3.12**; Python 3.14 is not used because PyTorch/Ultralytics support is unreliable there)
- About 16 GB RAM, CPU-only is expected
- Internet **once**, so Ultralytics can download `yolov8n.pt` (~6 MB) plus PyTorch wheels on first install

## Setup (Windows)

In PowerShell, from this folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If script activation is blocked:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Run

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

First launch may take several minutes while PyTorch/YOLO weights finish downloading.

## Suggested demo path

1. Click **Load demo records** in the sidebar (synthetic, labelled, not model output).
2. Review the dashboard cards and recent events.
3. Optionally register a real asset with a baseline photo.
4. On **Image monitoring**, upload a new photo of the **same** framing and run analysis.

## Privacy

- Images and the SQLite database stay under `data/` on this computer
- No cloud image APIs
- No facial recognition
- Human verification is required before acting on any alert

## Status meanings

| Status | Meaning |
| --- | --- |
| UNCHANGED / DETECTED | Category evidence and/or visual similarity suggest a stable view |
| POSSIBLE CHANGE | Mixed signals (angle, lighting, clutter, or a real change) |
| NOT DETECTED — VERIFY | Weak category evidence and/or a large visual difference — **review only**, never “confirmed theft” |

## Project layout

| Path | Role |
| --- | --- |
| `app.py` | Streamlit UI |
| `database.py` | SQLite assets and events |
| `vision.py` | YOLO + image comparison |
| `demo.py` | Labelled synthetic records |
| `paths.py` | Local folders |
| `data/` | Database, uploads, evidence (gitignored) |

## Limitations

- YOLOv8n/COCO has **no printer class**; printer checks lean on view comparison
- Category ≠ unique asset identity
- Same camera viewpoint is assumed
- Not a production security or audit system
- If YOLO cannot load, live detection is disabled; demo mode still works and the UI will say so
