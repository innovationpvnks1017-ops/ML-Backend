from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ml_processor import MLProcessor

# Initialize FastAPI app
app = FastAPI(
    title="ML Experiment API",
    description="Backend API for running ML experiments",
    version="1.0",
)

# Allow frontend (Lovable / Next.js) to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or restrict to your frontend domain
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------- #
#  Define API Schema
# ---------------------- #
class ExperimentRequest(BaseModel):
    dataset_name: str
    model_name: str
    test_size: float = 0.2
    random_state: int = 42


# ---------------------- #
#  API Route
# ---------------------- #
@app.post("/api/run-experiment")
async def run_experiment(request: ExperimentRequest):
    """Endpoint that runs ML pipeline and returns metrics."""
    try:
        processor = MLProcessor(
            dataset_name=request.dataset_name,
            model_name=request.model_name,
            test_size=request.test_size,
            random_state=request.random_state,
        )
        results = processor.run_experiment()
        return {"status": "success", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Root endpoint
@app.get("/")
async def root():
    return {"message": "Backend is running! Visit /api/run-experiment to test."}
