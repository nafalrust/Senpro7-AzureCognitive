from dataclasses import dataclass, field


@dataclass
class DetectedObject:
    name: str
    confidence: float
    x: int
    y: int
    width: int
    height: int


@dataclass
class AnalysisResult:
    image_path: str
    width: int
    height: int
    caption: str
    caption_confidence: float
    objects: list[DetectedObject] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    dense_captions: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "AnalysisResult":
        data = dict(data)
        data["objects"] = [DetectedObject(**obj) for obj in data.get("objects", [])]
        return cls(**data)
