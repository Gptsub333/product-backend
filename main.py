from fastapi import FastAPI
from router.complete import router as complete_router
from router.intent import router as intent_router
import subprocess
import sys

app = FastAPI()

# Include both API routers
app.include_router(complete_router, prefix="/api/response")


app.include_router(intent_router, prefix="/api/intent")


