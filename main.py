from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ml_processor import MLProcessor  # ✅ FIXED: no 'backend.'
  # ✅ correct import

app = FastAPI()  # ✅ this must be global, not inside a function

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schema
class ExperimentRequest(BaseModel):
    dataset_name: str
    model_name: str
    parameters: dict = {}

@app.post("/api/run-experiment")
async def run_experiment(request: ExperimentRequest):
    try:
        # Call your actual ML logic
        result = MLProcessor(
            dataset_name=request.dataset_name,
            model_name=request.model_name,
            parameters=request.parameters,
        )
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
