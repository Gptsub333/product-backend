from fastapi import FastAPI
from router.complete import router as complete_router
from router.intent import router as intent_router
from router.direct_chat import router as direct_chat_router
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
    # Allow specific HTTP methods
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    # Allow specific headers
    allow_headers=["X-Custom-Header", "Content-Type"],
)

# Include all API routers
app.include_router(complete_router, prefix="/api/response")
app.include_router(intent_router, prefix="/api/intent")
app.include_router(direct_chat_router, prefix="/api/chat")


@app.get("/")
async def root():
    return {
        "message": "Chatbot API is running",
        "endpoints": [
            "/api/response - For queries using RAG with embeddings",
            "/api/intent - For uploading documents and creating chatbots",
            "/api/chat - For direct paper-based QA"
        ]
    }
