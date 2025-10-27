"""Autonomous web search and indexing for the research site, powered by research_llm."""

import logging
import requests
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

# Import from research_llm module
try:
    from research_llm import AutonomousWebSearch, load_config, DEFAULTS
    WEB_SEARCH_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: research_llm or AutonomousWebSearch not available: {e}")
    WEB_SEARCH_AVAILABLE = False

# Import Google search service
try:
    from google_search import google_search_service
    GOOGLE_SEARCH_AVAILABLE = True
except ImportError:
    GOOGLE_SEARCH_AVAILABLE = False
    print("WARNING: Google search not available")

logger = logging.getLogger(__name__)

class SearchEngineService:
    """Service wrapper for AutonomousWebSearch functionality."""

    def __init__(self, config: Optional[Dict] = None):
        if not WEB_SEARCH_AVAILABLE:
            raise RuntimeError("Autonomous web search not available. Check imports.")

        self.config = config or load_config()

        # Set paths relative to Portfolio-Website-main/data
        base_dir = Path(__file__).parent.parent / "data"
        base_dir.mkdir(exist_ok=True)

        self.config["WEB_INDEX_PATH"] = str(base_dir / "web_index.db")
        
        self._web_search: Optional[AutonomousWebSearch] = None
        print("✓ Search Engine Service initialized")

    def _get_web_search(self) -> AutonomousWebSearch:
        """Get or create web search instance."""
        if self._web_search is None:
            self._web_search = AutonomousWebSearch(
                index_path=self.config["WEB_INDEX_PATH"],
                crawler_config={
                    "max_pages": int(self.config.get("WEB_CRAWLER_MAX_PAGES", DEFAULTS["WEB_CRAWLER_MAX_PAGES"])),
                    "max_depth": int(self.config.get("WEB_CRAWLER_MAX_DEPTH", DEFAULTS["WEB_CRAWLER_MAX_DEPTH"])),
                    "delay": float(self.config.get("WEB_CRAWLER_DELAY", DEFAULTS["WEB_CRAWLER_DELAY"]))
                }
            )
        return self._web_search

    def crawl(self, seed_urls: List[str], max_depth: int = 1) -> Dict:
        """Crawl and index websites."""
        if not WEB_SEARCH_AVAILABLE:
            return {"success": False, "error": "Web search functionality not available"}
        
        web_search = self._get_web_search()
        indexed_count = web_search.index_multiple_websites(
            seed_urls,
            max_pages=int(self.config.get("WEB_CRAWLER_MAX_PAGES", DEFAULTS["WEB_CRAWLER_MAX_PAGES"])),
            max_depth=max_depth
        )
        stats = web_search.get_stats()
        return {"success": True, "indexed": indexed_count, "total": stats.get("total_pages", 0)}

    def local_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search the local web index."""
        if not WEB_SEARCH_AVAILABLE:
            return []
        
        web_search = self._get_web_search()
        results = web_search.search(query, max_results=top_k)
        
        formatted_results = []
        for r in results:
            formatted_results.append({
                "url": r.url,
                "title": r.title,
                "snippet": r.snippet,
                "score": r.score,
                "timestamp": r.timestamp,
            })
        return formatted_results

    def get_stats(self) -> Dict:
        """Get web search statistics."""
        if not WEB_SEARCH_AVAILABLE:
            return {"total_pages": 0, "total_domains": 0, "index_size_mb": 0}

        web_search = self._get_web_search()
        stats = web_search.get_stats()
        return {
            "total_pages": stats.get("total_pages", 0),
            "total_domains": stats.get("total_domains", 0),
            "index_size_mb": stats.get("index_size_mb", 0)
        }

    def index_url(self, url: str, depth: int = 1, max_pages: int = 20, same_domain_only: bool = True) -> Dict:
        """Index a URL with depth=1 parsing."""
        if not WEB_SEARCH_AVAILABLE:
            return {"success": False, "error": "Web search not available"}

        try:
            # Ensure URL has scheme
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url

            web_search = self._get_web_search()

            # Fetch and parse the main page
            response = requests.get(url, timeout=20, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract title and text content
            title = soup.title.string.strip() if soup.title and soup.title.string else url
            text_content = soup.get_text(separator=' ', strip=True)

            # Create a snippet from the beginning of the text
            snippet = text_content[:300] + "..." if len(text_content) > 300 else text_content

            # Index the main page
            page_data = {
                "url": url,
                "title": title,
                "snippet": snippet,
                "body": text_content,
                "timestamp": str(response.headers.get('date', '')),
                "domain": urlparse(url).netloc
            }

            web_search.web_index.index_page(page_data)

            indexed_pages = 1
            indexed_domains = {urlparse(url).netloc}

            # If depth >= 1, find and index same-domain links
            if depth >= 1:
                links = []
                for a_tag in soup.find_all('a', href=True):
                    link = urljoin(url, a_tag['href'])
                    parsed_link = urlparse(link)

                    # Only index same domain if requested
                    if same_domain_only:
                        if parsed_link.netloc == urlparse(url).netloc:
                            links.append(link)
                    else:
                        links.append(link)

                # Remove duplicates and limit to max_pages
                links = list(set(links))[:max_pages]

                for link in links:
                    try:
                        # Skip if already indexed
                        if link == url:
                            continue

                        link_response = requests.get(link, timeout=15, headers={
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                        })
                        link_response.raise_for_status()

                        link_soup = BeautifulSoup(link_response.content, 'html.parser')
                        link_title = link_soup.title.string.strip() if link_soup.title and link_soup.title.string else link
                        link_text = link_soup.get_text(separator=' ', strip=True)
                        link_snippet = link_text[:300] + "..." if len(link_text) > 300 else link_text

                        link_page_data = {
                            "url": link,
                            "title": link_title,
                            "snippet": link_snippet,
                            "body": link_text,
                            "timestamp": str(link_response.headers.get('date', '')),
                            "domain": urlparse(link).netloc
                        }

                        web_search.web_index.index_page(link_page_data)
                        indexed_pages += 1
                        indexed_domains.add(urlparse(link).netloc)

                    except Exception as e:
                        logger.info(f"Skipped indexing link {link}: {e}")
                        continue

            return {
                "success": True,
                "indexed_pages": indexed_pages,
                "indexed_domains": list(indexed_domains),
                "main_url": url
            }

        except Exception as e:
            logger.error(f"Failed to index URL {url}: {e}")
            return {"success": False, "error": str(e)}

    def search_with_auto_indexing(self, query: str, num_results: int = 5, auto_index: bool = True, max_index_pages: int = 20, use_google: bool = True) -> Dict:
        """Search with automatic Google fallback and indexing."""
        try:
            local_results = []

            # First try local index
            if WEB_SEARCH_AVAILABLE:
                web_search = self._get_web_search()
                local_search_results = web_search.search(query, max_results=num_results)

                local_results = [{
                    "url": r.url,
                    "title": r.title,
                    "snippet": r.snippet,
                    "score": r.score,
                    "source": "local_index"
                } for r in local_search_results]

            google_results = []
            newly_indexed_domains = []

            # If we don't have enough local results and Google is enabled, use Google search
            if len(local_results) < num_results and use_google and GOOGLE_SEARCH_AVAILABLE:
                try:
                    # Perform Google search
                    google_search_result = google_search_service.search_and_process(query, num_results=5)
                    if google_search_result.get('success'):
                        google_data = google_search_result.get('results', [])

                        google_results = [{
                            "url": r.get('link', ''),
                            "title": r.get('title', ''),
                            "snippet": r.get('snippet', ''),
                            "score": r.get('relevance_score', 0.8),
                            "content_type": r.get('content_type', 'web'),
                            "source": "google"
                        } for r in google_data]

                        # Auto-index top Google results if requested
                        if auto_index:
                            for result in google_data[:3]:  # Index top 3 results
                                url = result.get('link', '')
                                if url:
                                    try:
                                        index_result = self.index_url(url, depth=1, max_pages=max_index_pages)
                                        if index_result.get('success'):
                                            newly_indexed_domains.extend(index_result.get('indexed_domains', []))
                                            logger.info(f"Successfully auto-indexed {url}")
                                    except Exception as e:
                                        logger.info(f"Skipped auto-indexing {url}: {str(e)[:100]}")

                except Exception as e:
                    logger.warning(f"Google search failed: {e}")

            # Get updated stats
            stats = self.get_stats()

            return {
                "success": True,
                "local_results": local_results,
                "google_results": google_results,
                "newly_indexed_domains": list(set(newly_indexed_domains)),
                "total_indexed_domains": stats.get("total_domains", 0),
                "total_indexed_pages": stats.get("total_pages", 0)
            }

        except Exception as e:
            logger.error(f"Search with auto-indexing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "local_results": [],
                "google_results": [],
                "newly_indexed_domains": [],
                "total_indexed_domains": 0,
                "total_indexed_pages": 0
            }

_search_engine_service_instance: Optional[SearchEngineService] = None

def get_search_service() -> SearchEngineService:
    global _search_engine_service_instance
    if _search_engine_service_instance is None:
        _search_engine_service_instance = SearchEngineService()
    return _search_engine_service_instance