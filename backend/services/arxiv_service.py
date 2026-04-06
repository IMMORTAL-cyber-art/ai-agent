import httpx
import xml.etree.ElementTree as ET
from models.schemas import PaperInfo
from typing import List
import logging

logger = logging.getLogger(__name__)

async def fetch_papers_from_arxiv(topic: str, limit: int = 5) -> List[PaperInfo]:
    """
    Search for papers on arXiv by topic.
    Extract title, summary, authors, and year.
    """
    url = "https://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{topic}",
        "start": 0,
        "max_results": limit,
        "sortBy": "relevance",
        "sortOrder": "descending"
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            
            # Parse Atom/XML
            root = ET.fromstring(response.text)
            
            # Atom XML namespace
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            papers = []
            for entry in root.findall('atom:entry', ns):
                title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
                summary = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
                authors = [author.find('atom:name', ns).text for author in entry.findall('atom:author', ns)]
                published = entry.find('atom:published', ns).text
                year = int(published[:4]) if published else None
                
                paper = PaperInfo(
                    title=title,
                    abstract=summary,
                    authors=authors,
                    year=year
                )
                papers.append(paper)
            
            logger.info(f"Successfully fetched {len(papers)} papers from arXiv.")
            return papers
            
    except Exception as e:
        logger.error(f"Error fetching from arXiv: {e}")
        return []
