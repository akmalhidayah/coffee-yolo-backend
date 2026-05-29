from typing import Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.routes.models import router as models_router
from app.routes.predict import router as predict_router

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router)
app.include_router(models_router)


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {
        "status": "ok",
        "message": "Coffee Quality API is running",
    }
