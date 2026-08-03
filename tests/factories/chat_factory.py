"""Test data builders for chat sessions and messages."""


def create_session(client, title="测试对话"):
    resp = client.post("/chat/new_session", json={"title": title})
    data = resp.get_json()
    return data.get("session_id") if data else None


def create_message(client, session_id, content="Hello"):
    resp = client.post("/chat/send", json={
        "session_id": session_id,
        "content": content,
    })
    return resp.get_json()
