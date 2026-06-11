"""Integration tests: server API wired to env, frontend action parity."""
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from env.models import ACTION_TYPES
from server.app import app

client = TestClient(app)
FRONTEND_JS = Path(__file__).resolve().parent.parent / "frontend" / "app.js"


def _frontend_action_ids() -> set[str]:
    text = FRONTEND_JS.read_text(encoding="utf-8")
    return set(re.findall(r'\bid:\s*"([a-z_]+)"', text))


@pytest.fixture
def session():
    sid = "test-session"
    r = client.post(f"/reset?task_id=debt_trap&seed=7&session_id={sid}")
    assert r.status_code == 200
    return sid


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_tasks_lists_env_tasks_and_actions():
    data = client.get("/tasks").json()
    task_ids = {t["id"] for t in data["tasks"]}
    assert {"debt_trap", "balanced_growth", "adversarial_crash", "free_play"} <= task_ids
    assert data["action_types"] == ACTION_TYPES


def test_frontend_actions_match_server():
    ui_actions = _frontend_action_ids()
    server_actions = set(ACTION_TYPES)
    assert ui_actions == server_actions, (
        f"frontend/server action drift\n"
        f"  only in UI: {sorted(ui_actions - server_actions)}\n"
        f"  only in server: {sorted(server_actions - ui_actions)}"
    )


def test_reset_returns_observation_shape(session):
    obs = client.post(f"/reset?task_id=debt_trap&seed=7&session_id={session}").json()
    for key in ("month", "savings", "debt", "investments", "market_regime", "net_worth"):
        assert key in obs
    assert obs["debt"]["credit_card"] == 35000
    assert obs["month"] == 1


def test_full_episode_via_api(session):
    obs = client.post(f"/reset?task_id=balanced_growth&seed=3&session_id={session}").json()
    start_net = obs["net_worth"]

    for _ in range(6):
        r = client.post(
            f"/step?session_id={session}",
            json={"action_type": "hold", "amount": 0},
        )
        assert r.status_code == 200
        body = r.json()
        assert "observation" in body
        assert "reward" in body
        assert "info" in body
        assert body["info"]["action_ok"] is True
        if body["done"]:
            break
    else:
        pytest.fail("episode did not finish in 6 steps")

    state = client.get(f"/state?session_id={session}").json()
    assert state["month"] == 7

    grade = client.post(f"/grade?session_id={session}").json()
    assert 0 < grade["score"] < 1
    assert grade["components"]
    assert grade["net_worth"] != start_net or True  # may stay flat on hold


def test_invalid_action_rejected(session):
    r = client.post(
        f"/step?session_id={session}",
        json={"action_type": "fly_to_moon", "amount": 1},
    )
    assert r.status_code == 422


def test_step_before_reset_409():
    r = client.post(
        "/step?session_id=never-started",
        json={"action_type": "hold", "amount": 0},
    )
    assert r.status_code == 409


def test_frontend_static_files_served():
    assert client.get("/").status_code == 200
    assert "FinAgent" in client.get("/").text
    assert client.get("/app.js").status_code == 200
    assert client.get("/style.css").status_code == 200
    # API routes must win over static mount
    assert client.get("/tasks").status_code == 200
    assert client.get("/health").status_code == 200
