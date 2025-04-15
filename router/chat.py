from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from modules.llm_selector import run

router = APIRouter()


class ChatRequest(BaseModel):
    userId: str
    chatbotName: str
    llm: str
    text: str


@router.post("/")
async def chat(request_data: ChatRequest):
    # Convert to the format expected by llm_selector
    input_data = {
        "userId": request_data.userId,
        "chatbotName": request_data.chatbotName,
        "llm": request_data.llm,
        "text": request_data.text
    }

    result = run(input_data)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result
