from pathlib import Path
from typing import List, Set

from pydantic import BaseModel


class Settings(BaseModel):
    app_name: str = "Coffee Quality YOLO API"
    app_version: str = "0.1.0"
    cors_origins: List[str] = ["*"]

    base_dir: Path = Path(__file__).resolve().parents[2]
    upload_dir: Path = base_dir / "uploads"
    model_path: Path = base_dir / "models" / "best.pt"

    allowed_extensions: Set[str] = {"jpg", "jpeg", "png"}


settings = Settings()
settings.upload_dir.mkdir(parents=True, exist_ok=True)
