import time
from pathlib import Path

from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError

import config
from models import AnalysisResult, DetectedObject

VISUAL_FEATURES = [VisualFeatures.CAPTION, VisualFeatures.OBJECTS, VisualFeatures.TAGS,
    VisualFeatures.DENSE_CAPTIONS,
]


def create_client() -> ImageAnalysisClient:
    return ImageAnalysisClient(
        endpoint=config.VISION_ENDPOINT,
        credential=AzureKeyCredential(config.VISION_KEY),
    )


def _parse_response(image_path: Path, response) -> AnalysisResult:
    caption = response.caption
    objects = response.objects.list if response.objects else []
    tags = response.tags.list if response.tags else []
    dense = response.dense_captions.list if response.dense_captions else []

    return AnalysisResult(
        image_path=str(image_path),
        width=response.metadata.width,
        height=response.metadata.height,
        caption=caption.text if caption else "(tidak ada caption)",
        caption_confidence=caption.confidence if caption else 0.0,
        objects=[
            DetectedObject(
                name=obj.tags[0].name,
                confidence=obj.tags[0].confidence,
                x=obj.bounding_box.x,
                y=obj.bounding_box.y,
                width=obj.bounding_box.width,
                height=obj.bounding_box.height,
            )
            for obj in objects
        ],
        tags=[tag.name for tag in tags],
        dense_captions=[
            item.text for item in dense
            if caption is None or item.text != caption.text
        ],
    )


def analyze_image(client: ImageAnalysisClient, image_path: Path) -> AnalysisResult:
    image_path = Path(image_path)
    image_bytes = image_path.read_bytes()

    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            response = client.analyze(
                image_data=image_bytes,
                visual_features=VISUAL_FEATURES,
                gender_neutral_caption=True,
            )
            return _parse_response(image_path, response)
        except HttpResponseError as error:
            if error.status_code == 429 and attempt < config.MAX_RETRIES:
                wait = 15 * attempt
                print(f"  ! rate limit, menunggu {wait} detik (percobaan {attempt})...")
                time.sleep(wait)
                continue
            raise
