import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "images"
OUTPUT_DIR = BASE_DIR / "outputs"
CACHE_FILE = OUTPUT_DIR / "results_cache.json"

IMAGES_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

VISION_KEY = os.getenv("VISION_KEY")
VISION_ENDPOINT = os.getenv("VISION_ENDPOINT")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
MIN_OBJECT_CONFIDENCE = 0.5
REQUEST_DELAY_SECONDS = 3.5  # aman untuk tier F0
MAX_RETRIES = 3


def new_run_dir() -> Path:
    run_dir = OUTPUT_DIR / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir()
    return run_dir


def validate() -> None:
    missing = [
        name
        for name, value in (("VISION_KEY", VISION_KEY), ("VISION_ENDPOINT", VISION_ENDPOINT))
        if not value
    ]
    if missing:
        raise SystemExit(f"Variabel berikut belum diisi di file .env: {', '.join(missing)}")
