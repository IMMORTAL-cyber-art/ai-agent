from fastapi import APIRouter, HTTPException
from models.schemas import ReviewRequest, GeneratedReviewResponse, QARequest, QAResponse
from services.scholar_service import fetch_papers_for_topic
from services.llm_service import generate_literature_review, answer_question
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/generate-review", response_model=GeneratedReviewResponse)
async def generate_review(request: ReviewRequest):
    """
    Given a topic, fetches relevance papers, and generates an AI literature review.
    """
    logger.info(f"Received request to generate review for topic: {request.topic} Language: {request.language}")
    
    # 1. Fetch papers (Try Semantic Scholar first)
    papers = []
    try:
        logger.info(f"Attempting to fetch papers from Semantic Scholar...")
        papers = await fetch_papers_for_topic(request.topic, limit=8)
    except Exception as e:
        logger.warning(f"Semantic Scholar failed: {e}. Trying arXiv fallback...")
        try:
            from services.arxiv_service import fetch_papers_from_arxiv
            papers = await fetch_papers_from_arxiv(request.topic, limit=8)
        except Exception as arxiv_err:
            logger.error(f"Both sources failed. arXiv error: {arxiv_err}")
            raise HTTPException(status_code=500, detail="Scholar is currently busy and the fallback (arXiv) also failed. Please try again in a few minutes.")
        
    if not papers:
        # One last try if Scholar returned empty but didn't error
        from services.arxiv_service import fetch_papers_from_arxiv
        papers = await fetch_papers_from_arxiv(request.topic, limit=8)
        
    if not papers:
        raise HTTPException(status_code=404, detail="No relevant papers were found for this topic on either Semantic Scholar or arXiv.")
        
    # 2. Generate review
    llm_result = await generate_literature_review(request.topic, papers, request.language)
    
    if isinstance(llm_result, dict) and "error" in llm_result:
        raise HTTPException(
            status_code=500, 
            detail=llm_result["error"]
        )
        
    return GeneratedReviewResponse(
        structured_review=llm_result["structured_review"],
        generation_time_seconds=llm_result["generation_time_seconds"],
        papers=papers
    )

@router.post("/qa", response_model=QAResponse)
async def ask_question(request: QARequest):
    """
    Allows asking questions interactively to the AI based on the topic.
    """
    logger.info(f"Received QA request for topic: {request.topic}")
    
    answer = await answer_question(request.topic, request.question, request.chat_history, request.language)
    
    return QAResponse(answer=answer)
