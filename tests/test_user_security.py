import pytest

from app.services import user_service
from app.services.token_service import authenticate_bearer_token, create_access_token


def test_duplicate_registration_does_not_overwrite_user(tmp_path, monkeypatch):
    monkeypatch.setattr(user_service.settings, "data_dir", tmp_path)
    monkeypatch.setattr(user_service.settings, "database_path", tmp_path / "users.db")
    first = user_service.register_user(name="Nama Awal", email="a@example.com", password="rahasia")
    with pytest.raises(user_service.DuplicateEmailError):
        user_service.register_user(name="Nama Baru", email="a@example.com", password="berubah")
    assert user_service.serialize_user(user_service.get_user_by_id(first["id"]))["name"] == "Nama Awal"


def test_google_login_preserves_existing_role_and_provider(tmp_path, monkeypatch):
    monkeypatch.setattr(user_service.settings, "data_dir", tmp_path)
    monkeypatch.setattr(user_service.settings, "database_path", tmp_path / "users.db")
    original = user_service.register_user(name="Admin", email="admin@example.com", password="x")
    with user_service._connect() as connection:
        connection.execute("UPDATE users SET role = 'admin' WHERE id = ?", (original["id"],))
        connection.commit()
    google = user_service.get_or_create_google_user(name="Nama Google", email="admin@example.com")
    assert google["role"] == "admin"
    assert google["auth_provider"] == "email"


def test_inactive_user_cannot_login_or_use_old_token(tmp_path, monkeypatch):
    monkeypatch.setattr(user_service.settings, "data_dir", tmp_path)
    monkeypatch.setattr(user_service.settings, "database_path", tmp_path / "users.db")
    user = user_service.register_user(name="User", email="u@example.com", password="secret")
    token = create_access_token(user)
    with user_service._connect() as connection:
        connection.execute("UPDATE users SET is_active = 0 WHERE id = ?", (user["id"],))
        connection.commit()
    with pytest.raises(user_service.InactiveAccountError):
        user_service.login_user(email="u@example.com", password="secret")
    with pytest.raises(user_service.InactiveAccountError):
        authenticate_bearer_token(f"Bearer {token}")


def test_mark_seen_does_not_reactivate_user(tmp_path, monkeypatch):
    monkeypatch.setattr(user_service.settings, "data_dir", tmp_path)
    monkeypatch.setattr(user_service.settings, "database_path", tmp_path / "users.db")
    user = user_service.register_user(name="User", email="seen@example.com", password="secret")
    with user_service._connect() as connection:
        connection.execute("UPDATE users SET is_active = 0 WHERE id = ?", (user["id"],))
        connection.commit()
    user_service.mark_user_seen(user["id"])
    assert user_service.get_user_by_id(user["id"])["is_active"] == 0


def test_profile_duplicate_preserves_sensitive_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(user_service.settings, "data_dir", tmp_path)
    monkeypatch.setattr(user_service.settings, "database_path", tmp_path / "users.db")
    first = user_service.register_user(name="First", email="first@example.com", password="one")
    second = user_service.register_user(name="Second", email="second@example.com", password="two")
    with pytest.raises(user_service.DuplicateEmailError):
        user_service.update_profile(
            user_id=first["id"], name="Changed", email=second["email"],
            location="New", phone="123",
        )
    stored = user_service.get_user_by_id(first["id"])
    assert stored["auth_provider"] == "email"
    assert stored["role"] == "user"
    assert stored["is_active"] == 1
