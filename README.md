# AI Literature Review Generator

A full-stack AI-powered tool for generating structured literature reviews from research topics.
It uses FastAPI and Semantic Scholar on the backend, and React/Tailwind CSS on the frontend.

## Prerequisites
- Node.js (v18+)
- Python 3 (v3.9+)
- OpenAI API Key

## Setup Backend (Python / FastAPI)
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Create and activate a Virtual Environment:
   ```bash
   python -m venv venv
   # On Windows
   .\venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Setup environment variables:
   Copy `.env.example` to `.env` and fill in your OpenAI key:
   ```bash
   OPENAI_API_KEY=sk-xxxx...
   ```
5. Run the dev server:
   ```bash
   uvicorn main:app --reload --port 8000
   ```
   *The backend will be running at http://localhost:8000*

## Setup Frontend (React / Vite)
1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Run the dev server:
   ```bash
   npm run dev
   ```
   *The frontend will be running at http://localhost:5173* (or similar port).

## Usage
1. Open the frontend URL in your browser.
2. Enter a research topic like "Quantum Computing in Healthcare".
3. Wait for the papers to be fetched and the OpenAI model to synthesize the literature.
4. Enjoy your structured literature review!
