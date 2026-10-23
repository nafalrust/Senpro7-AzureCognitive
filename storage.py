import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from models import AnalysisResult


def save_results(results: list[AnalysisResult], path: Path) -> None:
    data = [asdict(result) for result in results]
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_results(path: Path) -> list[AnalysisResult]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [AnalysisResult.from_dict(item) for item in data]


def results_to_dataframe(results: list[AnalysisResult], min_confidence: float) -> pd.DataFrame:
    rows = []
    for result in results:
        visible = [obj for obj in result.objects if obj.confidence >= min_confidence]
        rows.append(
            {
                "image": Path(result.image_path).name,
                "caption": result.caption,
                "caption_confidence": round(result.caption_confidence, 3),
                "n_objects": len(visible),
                "objects": ", ".join(sorted({obj.name for obj in visible})),
                "top_tags": ", ".join(result.tags[:5]),
            }
        )
    return pd.DataFrame(rows)


def save_summary_csv(results: list[AnalysisResult], path: Path, min_confidence: float) -> None:
    results_to_dataframe(results, min_confidence).to_csv(path, index=False)
