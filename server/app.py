from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from env.environment import FinAgentEnv
from env.models import Action, ACTION_TYPES
from env.tasks import TASKS, DEFAULT_TASK
from env.graders import grade_breakdown

app = FastAPI(title="FinAgentEnv", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# One environment per session so concurrent clients don't clobber each other.
_sessions: dict[str, FinAgentEnv] = {}
_session_tasks: dict[str, str] = {}


def _get_env(session_id: str) -> FinAgentEnv:
    env = _sessions.get(session_id)
    if env is None or env.state() == {}:
        raise HTTPException(status_code=409,
                            detail="No active episode for this session. Call /reset first.")
    return env


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/tasks")
async def list_tasks():
    out = []
    for task_id, task in {**TASKS, "free_play": DEFAULT_TASK}.items():
        out.append({
            "id": task_id,
            "name": task["name"],
            "difficulty": task["difficulty"],
            "description": task["description"],
            "goal": task["goal"],
            "initial": task["initial"],
        })
    return {"tasks": out, "action_types": ACTION_TYPES}


@app.post("/reset")
async def reset(task_id: str = None, seed: int = None,
                session_id: str = Query("default")):
    if task_id is not None and task_id not in TASKS and task_id != "free_play":
        raise HTTPException(status_code=422, detail=f"Unknown task_id '{task_id}'. "
                            f"Valid: {sorted(TASKS)} or 'free_play'.")
    env = _sessions.setdefault(session_id, FinAgentEnv())
    _session_tasks[session_id] = task_id or "free_play"
    obs = env.reset(task_id=task_id, seed=seed)
    return obs.model_dump()


@app.post("/step")
async def step(action: Action, session_id: str = Query("default")):
    if action.action_type not in ACTION_TYPES:
        raise HTTPException(status_code=422,
                            detail=f"Unknown action_type '{action.action_type}'. "
                                   f"Valid: {ACTION_TYPES}")
    env = _get_env(session_id)
    result = env.step(action)
    return {
        "observation": result.observation.model_dump(),
        "reward": result.reward,
        "done": result.done,
        "info": result.info.model_dump(),
    }


@app.get("/state")
async def get_state(session_id: str = Query("default")):
    env = _get_env(session_id)
    state = env.state()
    state["debt"] = state["debt"].model_dump()
    state["investments"] = state["investments"].model_dump()
    return state


@app.post("/grade")
async def grade_episode(session_id: str = Query("default")):
    env = _get_env(session_id)
    task_id = _session_tasks.get(session_id, "free_play")
    return grade_breakdown(task_id, env)


# Serve the dashboard (mounted last so API routes take priority).
_frontend = Path(__file__).resolve().parent.parent / "frontend"
if _frontend.is_dir():
    app.mount("/", StaticFiles(directory=str(_frontend), html=True), name="frontend")


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
