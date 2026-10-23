import time
from pathlib import Path

import config
import storage
import vision_service
from models import AnalysisResult


def list_images(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in Path(folder).iterdir()
        if path.suffix.lower() in config.IMAGE_EXTENSIONS
    )


def analyze_images(client, image_paths: list[Path], use_cache: bool = True) -> list[AnalysisResult]:
    cache = {}
    if use_cache:
        cache = {Path(r.image_path).name: r for r in storage.load_results(config.CACHE_FILE)}

    results = []
    total = len(image_paths)

    for index, path in enumerate(image_paths, start=1):
        cached = cache.get(path.name)
        if cached:
            cached.image_path = str(path)
            results.append(cached)
            print(f"[{index}/{total}] {path.name}  (dari cache)")
            continue

        print(f"[{index}/{total}] {path.name}  (memanggil Azure...)")
        try:
            result = vision_service.analyze_image(client, path)
        except Exception as error:
            print(f"  ! gagal menganalisis {path.name}: {error}")
            continue

        results.append(result)
        cache[path.name] = result
        storage.save_results(list(cache.values()), config.CACHE_FILE)
        time.sleep(config.REQUEST_DELAY_SECONDS)

    return results
