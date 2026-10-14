"""Local-only storage paths for IntelliWatch."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
BASELINES_DIR = UPLOADS_DIR / "baselines"
CURRENT_DIR = UPLOADS_DIR / "current"
EVIDENCE_DIR = DATA_DIR / "evidence"
DEMO_DIR = DATA_DIR / "demo_samples"
DB_PATH = DATA_DIR / "intelliwatch.db"

ALLOWED_IMAGE_TYPES = {"png", "jpg", "jpeg", "webp", "bmp"}


def ensure_directories() -> None:
    for folder in (DATA_DIR, UPLOADS_DIR, BASELINES_DIR, CURRENT_DIR, EVIDENCE_DIR, DEMO_DIR):
        folder.mkdir(parents=True, exist_ok=True)
