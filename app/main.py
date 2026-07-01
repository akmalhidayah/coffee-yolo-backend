from typing import Any, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.routes.admin import router as admin_router
from app.routes.auth import router as auth_router
from app.routes.models import router as models_router
from app.routes.predict import router as predict_router
from app.routes.users import router as users_router
from app.services.user_service import init_user_database

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
app.include_router(auth_router)
app.include_router(users_router)
app.include_router(admin_router)


@app.on_event("startup")
def startup() -> None:
    init_user_database()


@app.get("/health")
def health_check() -> Dict[str, Any]:
    return {
        "status": "ok",
        "message": "Coffee Quality API is running",
        "database": settings.database_path.exists(),
        "model": settings.model_path.exists(),
    }
