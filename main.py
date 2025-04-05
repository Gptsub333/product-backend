from fastapi import FastAPI
from router.complete import router as complete_router
from router.intent import router as intent_router
import subprocess
import sys
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ['*']
# Add CORS middleware to the FastAPI app
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow specific origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],  # Allow specific HTTP methods
    allow_headers=["X-Custom-Header", "Content-Type"],  # Allow specific headers
)

# Include both API routers
app.include_router(complete_router, prefix="/api/response")


app.include_router(intent_router, prefix="/api/intent")


