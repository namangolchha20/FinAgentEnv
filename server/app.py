import uvicorn
from fastapi import FastAPI, HTTPException
from env.environment import FinAgentEnv
from env.models import Observation, Action, StepResult

app = FastAPI(title="FinAgentEnv")
env = None

@app.on_event("startup")
def startup():
    global env
    env = FinAgentEnv()

@app.post("/reset")
async def reset(task_id: str = None, seed: int = None):
    global env
    obs = env.reset(task_id=task_id, seed=seed)
    return obs.dict()

@app.post("/step")
async def step(action: Action):
    global env
    result = env.step(action)
    return {"observation": result.observation.dict(), "reward": result.reward, "done": result.done, "info": result.info.dict()}

@app.get("/state")
async def get_state():
    global env
    return env.state()

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()