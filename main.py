from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ml_processor import MLProcessor  # ✅ Import your ML logic

# Initialize FastAPI app
app = FastAPI(
    title="ML Experiment API",
    description="Backend API for running ML experiments",
    version="1.0",
)

# ---------------------- #
#  CORS Setup
# ---------------------- #
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict later to your Lovable frontend domain
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------- #
#  Define Input Schema
# ---------------------- #
class ExperimentRequest(BaseModel):
    dataset: str                # From Lovable frontend: config.dataset
    models: list[str]           # From Lovable frontend: config.models
    learning_type: str | None = None
    visualizations: list[str] | None = None
    test_size: float = 0.2
    random_state: int = 42


# ---------------------- #
#  API Route
# ---------------------- #
@app.post("/api/run-experiment")
async def run_experiment(request: ExperimentRequest):
    """
    Endpoint that runs ML pipeline and returns evaluation metrics.
    Compatible with Lovable frontend config structure.
    """
    try:
        # Just run the first model (Lovable can send multiple)
        model_name = request.models[0] if request.models else "random_forest"

        processor = MLProcessor(
            dataset_name=request.dataset,
            model_name=model_name,
            test_size=request.test_size,
            random_state=request.random_state,
        )
        results = processor.run_experiment()

        return {"status": "success", "results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------- #
#  Root Endpoint
# ---------------------- #
@app.get("/")
async def root():
    return {"message": "✅ Backend is running! Use POST /api/run-experiment"}
