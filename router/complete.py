from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from modules.llm_selector import run

router = APIRouter()

class RequestData(BaseModel):
    llm: str
    text: str

@router.post("/")
async def generate_response(request_data: RequestData):
    input_data = request_data.dict()
    result = run(input_data)
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result
