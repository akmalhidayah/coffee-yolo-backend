import os
from pathlib import Path
from typing import List, Set

from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2]


def _parse_csv_env(value: str, default: List[str]) -> List[str]:
    if not value:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


def _path_env(key: str, default: Path) -> Path:
    raw_value = os.getenv(key, "")
    if not raw_value:
        return default
    path = Path(raw_value)
    if path.is_absolute():
        return path
    return BASE_DIR / path


class Settings(BaseModel):
    app_name: str = "Coffee Quality YOLO API"
    app_version: str = "0.1.0"
    cors_origins: List[str] = _parse_csv_env(
        os.getenv("CORS_ORIGINS", ""),
        ["http://localhost:3000", "http://localhost:5173", "http://localhost:8080"],
    )

    base_dir: Path = BASE_DIR
    upload_dir: Path = _path_env("UPLOAD_DIR", BASE_DIR / "uploads")
    data_dir: Path = BASE_DIR / "data"
    database_path: Path = data_dir / "coffee_yolo.db"
    model_path: Path = _path_env("MODEL_PATH", BASE_DIR / "models" / "best.pt")
    model_backup_path: Path = BASE_DIR / "models" / "best.previous.pt"
    google_client_id: str = os.getenv("GOOGLE_CLIENT_ID", "")
    jwt_secret: str = os.getenv(
        "JWT_SECRET_KEY",
        os.getenv("JWT_SECRET", "change-this-coffee-secret"),
    )
    jwt_algorithm: str = os.getenv("JWT_ALGORITHM", "HS256")
    access_token_expire_minutes: int = int(
        os.getenv(
            "ACCESS_TOKEN_EXPIRE_MINUTES",
            str(int(os.getenv("JWT_EXP_SECONDS", "86400")) // 60),
        )
    )
    admin_email: str = os.getenv("ADMIN_EMAIL", "").strip().lower()
    admin_password: str = os.getenv("ADMIN_PASSWORD", "")
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
    max_image_size_mb: int = int(os.getenv("MAX_IMAGE_SIZE_MB", "20"))
    yolo_image_size: int = int(os.getenv("YOLO_IMAGE_SIZE", "640"))
    yolo_iou_threshold: float = float(os.getenv("YOLO_IOU_THRESHOLD", "0.70"))
    yolo_max_detections: int = int(os.getenv("YOLO_MAX_DETECTIONS", "300"))
    yolo_device: str = os.getenv("YOLO_DEVICE", "cpu").strip() or "cpu"
    max_model_size_mb: int = int(os.getenv("MAX_MODEL_SIZE_MB", "200"))
    online_user_window_minutes: int = int(os.getenv("ONLINE_USER_WINDOW_MINUTES", "10"))

    allowed_extensions: Set[str] = {"jpg", "jpeg", "png"}
    allowed_model_extensions: Set[str] = {"pt"}

    @property
    def jwt_exp_seconds(self) -> int:
        return self.access_token_expire_minutes * 60

    @property
    def max_image_size_bytes(self) -> int:
        return self.max_image_size_mb * 1024 * 1024

    @property
    def max_model_size_bytes(self) -> int:
        return self.max_model_size_mb * 1024 * 1024


settings = Settings()
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.data_dir.mkdir(parents=True, exist_ok=True)
