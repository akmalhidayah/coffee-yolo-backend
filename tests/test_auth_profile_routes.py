import pytest
from fastapi import HTTPException

from app.routes import auth, users
from app.services import user_service


@pytest.fixture
def isolated_database(tmp_path, monkeypatch):
    monkeypatch.setattr(user_service.settings, "data_dir", tmp_path)
    monkeypatch.setattr(user_service.settings, "database_path", tmp_path / "users.db")


def test_duplicate_register_returns_409(isolated_database):
    payload = auth.RegisterRequest(name="User", email="dup@example.com", password="secret")
    auth.register(payload)
    with pytest.raises(HTTPException) as caught:
        auth.register(payload)
    assert caught.value.status_code == 409
    assert caught.value.detail == "Email sudah terdaftar."


def test_duplicate_profile_email_returns_409(isolated_database):
    first = user_service.register_user(name="First", email="first@example.com", password="one")
    user_service.register_user(name="Second", email="second@example.com", password="two")
    payload = users.ProfileRequest(
        name="First", email="second@example.com", location="Mamasa", phone=""
    )
    with pytest.raises(HTTPException) as caught:
        users.save_profile(payload, current_user=first)
    assert caught.value.status_code == 409
    assert caught.value.detail == "Email sudah digunakan oleh akun lain."


def test_profile_request_does_not_accept_auth_provider(isolated_database):
    payload = users.ProfileRequest(
        name="User", email="user@example.com", location="Mamasa",
        auth_provider="google",
    )
    assert "auth_provider" not in payload.dict()
