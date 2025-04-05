from fastapi import FastAPI
from router.pipeline import router as pipeline_router
from router.intent import router as intent_router

app = FastAPI()

# Include both API routers
app.include_router(pipeline_router, prefix="/api/response")
app.include_router(intent_router, prefix="/api/intent")
