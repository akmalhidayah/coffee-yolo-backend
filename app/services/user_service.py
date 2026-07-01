import hashlib
import hmac
import json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from passlib.context import CryptContext

from app.core.config import settings

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def init_user_database() -> None:
    with _connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL DEFAULT '',
                location TEXT NOT NULL DEFAULT '',
                phone TEXT NOT NULL DEFAULT '',
                auth_provider TEXT NOT NULL DEFAULT 'email',
                language TEXT NOT NULL DEFAULT 'Indonesia',
                role TEXT NOT NULL DEFAULT 'user',
                is_active INTEGER NOT NULL DEFAULT 1,
                last_login_at TEXT,
                last_seen_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        _ensure_user_columns(connection)
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                image_filename TEXT,
                status TEXT NOT NULL,
                class_name TEXT,
                coffee_type TEXT,
                grade TEXT,
                confidence REAL,
                bounding_boxes TEXT,
                detections TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        _seed_initial_admin(connection)
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
    role = existing["role"] if existing else "user"

    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO users (
                id, name, email, password_hash, location, phone,
                auth_provider, language, role, is_active, last_login_at,
                last_seen_at, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(email) DO UPDATE SET
                name = excluded.name,
                password_hash = CASE
                    WHEN excluded.password_hash = '' THEN users.password_hash
                    ELSE excluded.password_hash
                END,
                location = excluded.location,
                phone = excluded.phone,
                auth_provider = excluded.auth_provider,
                role = users.role,
                is_active = 1,
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
                role,
                1,
                existing["last_login_at"] if existing else None,
                existing["last_seen_at"] if existing else None,
                existing["created_at"] if existing else now,
                now,
            ),
        )
        connection.commit()

    return serialize_user(get_user_by_email(normalized_email))


def login_user(*, email: str, password: str = "") -> Optional[Dict[str, Any]]:
    init_user_database()
    user = get_user_by_email(email.strip().lower())
    if user is None or not user["password_hash"]:
        return None

    verified, needs_upgrade = _verify_password(password, user["password_hash"])
    if not verified:
        return None

    now = _now()
    new_hash = _hash_password(password) if needs_upgrade else user["password_hash"]
    with _connect() as connection:
        connection.execute(
            """
            UPDATE users
            SET password_hash = ?, last_login_at = ?, last_seen_at = ?,
                is_active = 1, updated_at = ?
            WHERE id = ?
            """,
            (new_hash, now, now, now, user["id"]),
        )
        connection.commit()

    return serialize_user(get_user_by_email(user["email"]))


def update_profile(
    *,
    name: str,
    email: str,
    location: str,
    phone: str = "",
    auth_provider: str = "email",
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    init_user_database()
    now = _now()
    with _connect() as connection:
        if user_id:
            connection.execute(
                """
                UPDATE users
                SET name = ?, email = ?, location = ?, phone = ?,
                    auth_provider = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    name.strip(),
                    email.strip().lower(),
                    location.strip(),
                    phone.strip(),
                    auth_provider.strip() or "email",
                    now,
                    user_id,
                ),
            )
            connection.commit()
            return serialize_user(get_user_by_id(user_id))

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


def get_user_by_id(user_id: str) -> Optional[sqlite3.Row]:
    with _connect() as connection:
        return connection.execute(
            "SELECT * FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()


def mark_user_seen(user_id: str) -> None:
    now = _now()
    with _connect() as connection:
        connection.execute(
            "UPDATE users SET last_seen_at = ?, is_active = 1, updated_at = ? WHERE id = ?",
            (now, now, user_id),
        )
        connection.commit()


def mark_user_login(user_id: str) -> Dict[str, Any]:
    now = _now()
    with _connect() as connection:
        connection.execute(
            """
            UPDATE users
            SET last_login_at = ?, last_seen_at = ?, is_active = 1, updated_at = ?
            WHERE id = ?
            """,
            (now, now, now, user_id),
        )
        connection.commit()
    return serialize_user(get_user_by_id(user_id))


def list_users_for_admin() -> List[Dict[str, Any]]:
    init_user_database()
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT
                users.*,
                COUNT(predictions.id) AS total_predictions
            FROM users
            LEFT JOIN predictions ON predictions.user_id = users.id
            GROUP BY users.id
            ORDER BY users.created_at DESC
            """
        ).fetchall()

    return [_serialize_admin_user(row) for row in rows]


def record_prediction_history(
    *,
    user_id: Optional[str],
    image_filename: str,
    response_status: str,
    prediction: Dict[str, Any],
) -> None:
    init_user_database()
    detections = prediction.get("detections") or []
    best_detection = (
        max(detections, key=lambda detection: detection.get("confidence", 0))
        if detections
        else {}
    )
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO predictions (
                user_id, image_filename, status, class_name, coffee_type, grade,
                confidence, bounding_boxes, detections, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                image_filename,
                response_status,
                best_detection.get("class_name") if response_status == "detected" else None,
                best_detection.get("coffee_type") if response_status == "detected" else None,
                best_detection.get("grade") if response_status == "detected" else None,
                best_detection.get("confidence") if response_status == "detected" else None,
                json.dumps(prediction.get("bounding_boxes") or [], ensure_ascii=False),
                json.dumps(detections, ensure_ascii=False),
                _now(),
            ),
        )
        connection.commit()


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
        "role": user["role"],
        "is_active": bool(user["is_active"]),
        "last_login_at": user["last_login_at"],
        "last_seen_at": user["last_seen_at"],
        "created_at": user["created_at"],
        "updated_at": user["updated_at"],
    }


def _connect() -> sqlite3.Connection:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(settings.database_path)
    connection.row_factory = sqlite3.Row
    return connection


def _ensure_user_columns(connection: sqlite3.Connection) -> None:
    columns = {
        row["name"]
        for row in connection.execute("PRAGMA table_info(users)").fetchall()
    }
    migrations = {
        "location": "ALTER TABLE users ADD COLUMN location TEXT NOT NULL DEFAULT ''",
        "phone": "ALTER TABLE users ADD COLUMN phone TEXT NOT NULL DEFAULT ''",
        "auth_provider": "ALTER TABLE users ADD COLUMN auth_provider TEXT NOT NULL DEFAULT 'email'",
        "language": "ALTER TABLE users ADD COLUMN language TEXT NOT NULL DEFAULT 'Indonesia'",
        "role": "ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'user'",
        "is_active": "ALTER TABLE users ADD COLUMN is_active INTEGER NOT NULL DEFAULT 1",
        "last_login_at": "ALTER TABLE users ADD COLUMN last_login_at TEXT",
        "last_seen_at": "ALTER TABLE users ADD COLUMN last_seen_at TEXT",
        "created_at": "ALTER TABLE users ADD COLUMN created_at TEXT NOT NULL DEFAULT ''",
        "updated_at": "ALTER TABLE users ADD COLUMN updated_at TEXT NOT NULL DEFAULT ''",
    }
    for column, statement in migrations.items():
        if column not in columns:
            connection.execute(statement)


def _seed_initial_admin(connection: sqlite3.Connection) -> None:
    if not settings.admin_email or not settings.admin_password:
        return
    admin_exists = connection.execute(
        "SELECT id FROM users WHERE role = 'admin' LIMIT 1"
    ).fetchone()
    if admin_exists:
        return

    now = _now()
    admin_id = str(uuid.uuid4())
    connection.execute(
        """
        INSERT INTO users (
            id, name, email, password_hash, location, phone, auth_provider,
            language, role, is_active, last_login_at, last_seen_at,
            created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(email) DO UPDATE SET
            password_hash = excluded.password_hash,
            role = 'admin',
            is_active = 1,
            updated_at = excluded.updated_at
        """,
        (
            admin_id,
            "Admin",
            settings.admin_email,
            _hash_password(settings.admin_password),
            "",
            "",
            "email",
            "Indonesia",
            "admin",
            1,
            None,
            None,
            now,
            now,
        ),
    )


def _hash_password(password: str) -> str:
    if not password:
        return ""
    return _pwd_context.hash(password)


def _verify_password(password: str, stored_hash: str) -> Tuple[bool, bool]:
    if not password or not stored_hash:
        return False, False
    if _is_sha256_hash(stored_hash):
        legacy_hash = hashlib.sha256(password.encode("utf-8")).hexdigest()
        return hmac_compare(legacy_hash, stored_hash), True
    try:
        return _pwd_context.verify(password, stored_hash), False
    except Exception:
        return False, False


def hmac_compare(left: str, right: str) -> bool:
    return hmac.compare_digest(left, right)


def _is_sha256_hash(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value.lower())


def _serialize_admin_user(user: sqlite3.Row) -> Dict[str, Any]:
    serialized = serialize_user(user)
    serialized["is_online"] = _is_online(user["last_seen_at"])
    serialized["total_predictions"] = int(user["total_predictions"] or 0)
    return serialized


def _is_online(last_seen_at: Optional[str]) -> bool:
    if not last_seen_at:
        return False
    try:
        seen_at = datetime.fromisoformat(last_seen_at)
    except ValueError:
        return False
    if seen_at.tzinfo is None:
        seen_at = seen_at.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - seen_at <= timedelta(
        minutes=settings.online_user_window_minutes
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
