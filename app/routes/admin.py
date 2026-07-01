from typing import Any, Dict

from fastapi import APIRouter, Depends

from app.services.token_service import get_current_admin_user
from app.services.user_service import list_users_for_admin

router = APIRouter(prefix="/admin", tags=["Admin"])


@router.get("/users")
def list_users(
    current_admin: Dict[str, Any] = Depends(get_current_admin_user),
) -> dict:
    return {
        "status": "success",
        "users": list_users_for_admin(),
    }
