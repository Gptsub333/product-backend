from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from modules.llm_selector import run

router = APIRouter()

class IntentRequestData(BaseModel):
    llm: str
    text: str

@router.post("/")
async def query_intent(request_data: IntentRequestData):
    input_data = request_data.dict()
    
    # Intent processing logic can be added here if you have a model for intent recognition
    # For now, it's the same as a complete response request for simplicity
    
    result = run(input_data)
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result
