from pydantic import BaseModel
from typing import List, Optional

class ReviewRequest(BaseModel):
    topic: str
    language: str = "English"

class ChatMessage(BaseModel):
    role: str
    content: str

class QARequest(BaseModel):
    topic: str
    question: str
    chat_history: List[ChatMessage] = []
    language: str = "English"

class QAResponse(BaseModel):
    answer: str

class PaperInfo(BaseModel):
    title: str
    abstract: Optional[str] = None
    authors: List[str] = []
    year: Optional[int] = None

class StructuredReview(BaseModel):
    introduction: str
    key_themes: List[str]
    comparative_analysis: str
    research_gaps: List[str]
    conclusion: str
    key_takeaways: List[str]
    ai_idea: str
    confidence_level: str
    complexity_level: str

class GeneratedReviewResponse(BaseModel):
    structured_review: StructuredReview
    papers: List[PaperInfo]
    generation_time_seconds: Optional[float] = None
