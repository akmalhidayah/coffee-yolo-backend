import hashlib
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from app.core.config import settings


def init_user_database() -> None:
    with _connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL DEFAULT '',
                location TEXT NOT NULL,
                phone TEXT NOT NULL DEFAULT '',
                auth_provider TEXT NOT NULL DEFAULT 'email',
                language TEXT NOT NULL DEFAULT 'Indonesia',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.commit()


def register_user(
    *,
    name: str,
    email: str,
    password: str = "",
    location: str = "Desa Masewe, Mamasa",
    phone: str = "",
    auth_provider: str = "email",
) -> Dict[str, Any]:
    init_user_database()
    now = _now()
    normalized_email = email.strip().lower()
    existing = get_user_by_email(normalized_email)
    user_id = existing["id"] if existing else str(uuid.uuid4())
    password_hash = _hash_password(password)

    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO users (
                id, name, email, password_hash, location, phone,
                auth_provider, language, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(email) DO UPDATE SET
                name = excluded.name,
                password_hash = CASE
                    WHEN excluded.password_hash = '' THEN users.password_hash
                    ELSE excluded.password_hash
                END,
                location = excluded.location,
                phone = excluded.phone,
                auth_provider = excluded.auth_provider,
                updated_at = excluded.updated_at
            """,
            (
                user_id,
                name.strip(),
                normalized_email,
                password_hash,
                location.strip(),
                phone.strip(),
                auth_provider.strip() or "email",
                existing["language"] if existing else "Indonesia",
                existing["created_at"] if existing else now,
                now,
            ),
        )
        connection.commit()

    return serialize_user(get_user_by_email(normalized_email))


def login_user(*, email: str, password: str = "") -> Optional[Dict[str, Any]]:
    init_user_database()
    user = get_user_by_email(email.strip().lower())
    if user is None:
        return None

    if not user["password_hash"]:
        return None

    if not password or user["password_hash"] != _hash_password(password):
        return None

    return serialize_user(user)


def update_profile(
    *,
    name: str,
    email: str,
    location: str,
    phone: str = "",
    auth_provider: str = "email",
) -> Dict[str, Any]:
    return register_user(
        name=name,
        email=email,
        password="",
        location=location,
        phone=phone,
        auth_provider=auth_provider,
    )


def get_user_by_email(email: str) -> Optional[sqlite3.Row]:
    with _connect() as connection:
        return connection.execute(
            "SELECT * FROM users WHERE email = ?",
            (email.strip().lower(),),
        ).fetchone()


def serialize_user(user: Optional[sqlite3.Row]) -> Dict[str, Any]:
    if user is None:
        return {}
    return {
        "id": user["id"],
        "name": user["name"],
        "email": user["email"],
        "location": user["location"],
        "phone": user["phone"],
        "auth_provider": user["auth_provider"],
        "language": user["language"],
        "created_at": user["created_at"],
        "updated_at": user["updated_at"],
    }


def _connect() -> sqlite3.Connection:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(settings.database_path)
    connection.row_factory = sqlite3.Row
    return connection


def _hash_password(password: str) -> str:
    if not password:
        return ""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
