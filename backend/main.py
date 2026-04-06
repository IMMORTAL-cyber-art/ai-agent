from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import review
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

# CORS configuration: allow localhost for dev + Vercel frontend URL for prod
FRONTEND_URL = os.getenv("FRONTEND_URL", "")
origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "https://*.vercel.app",
]
if FRONTEND_URL:
    origins.append(FRONTEND_URL)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,   # permissive for now; tighten to 'origins' after first deploy
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(review.router, prefix="/api", tags=["review"])

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI Literature Review Generator API"}
