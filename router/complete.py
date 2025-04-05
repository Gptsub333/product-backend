from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from modules.llm_selector import run
from typing import Optional

router = APIRouter()


class RequestData(BaseModel):
    llm: str
    text: str
    userId: Optional[str] = None
    botName: Optional[str] = None


@router.post("/")
async def generate_response(request_data: RequestData):
    input_data = request_data.dict()

    # Add userId and botName if available in the request
    if hasattr(request_data, "userId") and hasattr(request_data, "botName"):
        input_data["userId"] = request_data.userId
        input_data["botName"] = request_data.botName

    result = run(input_data)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result
