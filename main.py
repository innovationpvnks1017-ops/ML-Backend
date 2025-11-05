from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ml_processor import MLProcessor

app = FastAPI(
    title="ML Experiment API",
    description="Backend for Lovable ML Experiments",
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ExperimentRequest(BaseModel):
    dataset: str
    models: list[str]
    learning_type: str | None = None
    visualizations: list[str] | None = None
    test_size: float = 0.2
    random_state: int = 42


@app.post("/api/run-experiment")
async def run_experiment(request: ExperimentRequest):
    try:
        model_name = request.models[0] if request.models else "random-forest"
        processor = MLProcessor(
            dataset_name=request.dataset,
            model_name=model_name,
            test_size=request.test_size,
            random_state=request.random_state,
            visualizations=request.visualizations,
        )
        results = processor.run_experiment()
        return {"status": "success", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "✅ Backend running. Use POST /api/run-experiment"}
