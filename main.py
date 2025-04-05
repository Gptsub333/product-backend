from fastapi import FastAPI
from router.complete import router as complete_router
from router.intent import router as intent_router
import subprocess
import sys

app = FastAPI()

# Include both API routers
app.include_router(complete_router, prefix="/api/response")


app.include_router(intent_router, prefix="/api/intent")

# After the intent router is hit, call setup.py with the arguments
username = "user123"  # This will be replaced with actual variable
chatbotname = "bot123"  # This will be replaced with actual variable
file_location = "/path/to/file.pdf"  # This will be replaced with actual variable

# Run the setup.py script with the arguments
subprocess.run([sys.executable, "setup.py",
               username, chatbotname, file_location])
