"""Integration tests for auth routes — register, login, check_auth.

Requires PostgreSQL (clean_tables runs between tests).
"""
import pytest


pytestmark = pytest.mark.db


class TestCreateAccount:
    ROUTE = "/create_account"

    def test_success(self, client):
        resp = client.post(self.ROUTE, json={"username": "testuser", "pin": "123456"})
        data = resp.get_json()
        assert resp.status_code == 200
        assert data["success"] is True
        assert data["username"] == "testuser"

    def test_empty_username(self, client):
        resp = client.post(self.ROUTE, json={"username": "", "pin": "123456"})
        assert resp.status_code == 400

    def test_empty_pin(self, client):
        resp = client.post(self.ROUTE, json={"username": "user12345", "pin": ""})
        assert resp.status_code == 400

    def test_username_too_short(self, client):
        resp = client.post(self.ROUTE, json={"username": "abc", "pin": "123456"})
        assert resp.status_code == 400

    def test_username_too_long(self, client):
        resp = client.post(self.ROUTE, json={"username": "a" * 20, "pin": "123456"})
        assert resp.status_code == 400

    def test_invalid_pin_length_4_ok(self, client):
        resp = client.post(self.ROUTE, json={"username": "pintest4", "pin": "1234", "pin_length": 4})
        assert resp.status_code == 200

    def test_invalid_pin_length_not_4_or_6(self, client):
        resp = client.post(self.ROUTE, json={"username": "pintest5", "pin": "12345", "pin_length": 5})
        assert resp.status_code == 400

    def test_pin_non_digit(self, client):
        resp = client.post(self.ROUTE, json={"username": "pintest6", "pin": "12a456"})
        assert resp.status_code == 400

    def test_duplicate_username(self, client):
        client.post(self.ROUTE, json={"username": "dupuser", "pin": "123456"})
        resp = client.post(self.ROUTE, json={"username": "dupuser", "pin": "654321"})
        assert resp.status_code == 409

    def test_sets_session_variables(self, client):
        resp = client.post(self.ROUTE, json={"username": "sessiontest", "pin": "123456"})
        assert resp.status_code == 200
        resp2 = client.get("/check_auth")
        data = resp2.get_json()
        assert data["success"] is True
        assert data["authenticated"] is True
        assert data["username"] == "sessiontest"

    def test_pin_length_6_default(self, client):
        resp = client.post(self.ROUTE, json={"username": "pinlen6", "pin": "123456"})
        assert resp.status_code == 200


class TestLogin:
    ROUTE = "/login"

    def test_success(self, client):
        client.post("/create_account", json={"username": "logintest", "pin": "123456"})
        resp = client.post(self.ROUTE, json={"username": "logintest", "pin": "123456"})
        data = resp.get_json()
        assert resp.status_code == 200
        assert data["success"] is True
        assert data["username"] == "logintest"

    def test_wrong_pin(self, client):
        client.post("/create_account", json={"username": "wrongpin", "pin": "123456"})
        resp = client.post(self.ROUTE, json={"username": "wrongpin", "pin": "000000"})
        assert resp.status_code == 401

    def test_nonexistent_user(self, client):
        resp = client.post(self.ROUTE, json={"username": "nobody", "pin": "123456"})
        assert resp.status_code == 401

    def test_empty_credentials(self, client):
        resp = client.post(self.ROUTE, json={"username": "", "pin": ""})
        assert resp.status_code == 400

    def test_sets_session(self, client):
        client.post("/create_account", json={"username": "sessionlogin", "pin": "123456"})
        client.post(self.ROUTE, json={"username": "sessionlogin", "pin": "123456"})
        resp = client.get("/check_auth")
        data = resp.get_json()
        assert data["authenticated"] is True
        assert data["username"] == "sessionlogin"


class TestAdminLogin:
    ROUTE = "/login"

    def test_admin_login_success(self, client):
        resp = client.post(self.ROUTE, json={"username": "CEO", "pin": "123456"})
        data = resp.get_json()
        assert resp.status_code == 200
        assert data["is_admin"] is True
        assert data["is_auditor"] is True

    def test_admin_login_wrong_pin(self, client):
        resp = client.post(self.ROUTE, json={"username": "CEO", "pin": "wrong"})
        assert resp.status_code == 401

    def test_admin_login_sets_admin_role(self, client):
        client.post(self.ROUTE, json={"username": "CEO", "pin": "123456"})
        resp = client.get("/check_auth")
        data = resp.get_json()
        assert data["role"] == "admin"


class TestCheckAuth:
    ROUTE = "/check_auth"

    def test_authenticated(self, client):
        client.post("/create_account", json={"username": "checkme", "pin": "123456"})
        resp = client.get(self.ROUTE)
        data = resp.get_json()
        assert data["success"] is True
        assert data["authenticated"] is True
        assert data["username"] == "checkme"

    def test_not_authenticated_returns_success_with_flag(self, client):
        resp = client.get(self.ROUTE)
        data = resp.get_json()
        assert data["success"] is True
        assert data["authenticated"] is False

    def test_returns_user_id(self, client):
        client.post("/create_account", json={"username": "useridtest", "pin": "123456"})
        resp = client.get(self.ROUTE)
        data = resp.get_json()
        assert "user_id" in data
        assert len(data["user_id"]) > 10
