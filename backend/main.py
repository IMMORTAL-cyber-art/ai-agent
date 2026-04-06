from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routes import review
import logging
import os
from dotenv import load_dotenv

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="AI Literature Review Generator API",
    description="API for generating comprehensive literature reviews using Semantic Scholar and Gemini.",
    version="1.0.0"
)

# ── CORS CONFIGURATION (Fully Permissive for Deployment) ───────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ──────────────────────────────────────────────────────────────────────────────


app.include_router(review.router, prefix="/api", tags=["review"])

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI Literature Review Generator API"}
