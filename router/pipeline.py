from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from executor.executor import run_pipeline

router = APIRouter()

class ModuleCall(BaseModel):
    id: str
    type: str
    params: dict
    next: str = None

@router.post("/run-pipeline")
async def run_pipeline_api(modules: list[ModuleCall]):
    try:
        result = run_pipeline(modules)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
