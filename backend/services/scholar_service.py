from backend.models.schemas import PaperInfo
from typing import List
import logging
import os

logger = logging.getLogger(__name__)

async def fetch_papers_for_topic(topic: str, limit: int = 10) -> List[PaperInfo]:
    """
    Search for papers on Semantic Scholar by topic.
    Extract title, abstract, authors, and year.
    """
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    
    # Sanitize topic (lowercase, strip whitespace, remove quotes)
    clean_topic = topic.lower().strip().replace('"', '').replace("'", "")
    
    params = {
        "query": clean_topic,
        "limit": limit,
        "fields": "title,abstract,authors,year"
    }

    headers = {}
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key

    try:
        logger.info(f"🔍 Searching Semantic Scholar for: '{clean_topic}'")
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            papers = []
            for item in data.get("data", []):
                authors = [author.get("name") for author in item.get("authors", [])]
                paper = PaperInfo(
                    title=item.get("title", "Unknown Title"),
                    abstract=item.get("abstract"),
                    authors=authors,
                    year=item.get("year")
                )
                papers.append(paper)
            
            logger.info(f"✅ Semantic Scholar returned {len(papers)} results.")
            return papers
    except httpx.HTTPStatusError as e:
        logger.error(f"Semantic Scholar HTTP status error: {e.response.status_code} - {e}")
        # Re-raise so the router can handle the 429
        raise
    except Exception as e:
        logger.error(f"Error fetching from Semantic Scholar: {e}")
        return []
