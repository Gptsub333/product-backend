from fastapi import FastAPI
from router.pipeline import router as pipeline_router

app = FastAPI()
app.include_router(pipeline_router, prefix="/api")
