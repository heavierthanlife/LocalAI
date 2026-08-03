"""Test data builders for user accounts."""


def register_and_login(client, username="testuser", pin="123456"):
    """Register a user and log them in, returning the client."""
    client.post("/create_account", json={"username": username, "pin": pin, "pin_length": len(pin)})
    client.post("/login", json={"username": username, "pin": pin})
    return client


def make_admin(client, username="admin", pin="admin1234"):
    """Create an admin user by calling register (admin PIN must be set in env)."""
    import os
    os.environ.setdefault("ADMIN_PIN", pin)
    client.post("/create_account", json={"username": username, "pin": pin})
    client.post("/login", json={"username": username, "pin": pin})
    return client
