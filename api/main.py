from fastapi import FastAPI
from pydantic import BaseModel
from trinity_core import run_trinity_loop

# Create the FastAPI app instance
app = FastAPI(title="Trinity Mind API")

# Define what the API expects as input
class TrinityRequest(BaseModel):
    topic: str
    goal: str = "clarity"
    constraints: str = "realistic"

# POST endpoint that runs the Trinity reasoning loop
@app.post("/trinity/reason")
def reason(req: TrinityRequest):
    result = run_trinity_loop(req.topic, req.goal, req.constraints)
    return result

# Optional health check route
@app.get("/")
def root():
    return {"message": "Trinity Mind API is running"}
