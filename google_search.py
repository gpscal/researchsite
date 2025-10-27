import os
import requests
import logging
import re
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class GoogleSearchService:
    """Enhanced Google Custom Search API service with improved result processing"""
    
    def __init__(self):
        self.api_key = os.getenv('GOOGLE_API_KEY', 'AIzaSyDWRKgMbkn4G4imT7R5SjJlsUQUegfMQZ4')
        self.cx = os.getenv('GOOGLE_CX', 'e4fb3a57eba014f07')
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def search(self, query: str, num_results: int = 5, search_type: str = "web") -> Optional[Dict]:
        """Perform Google search using Custom Search API with enhanced query processing"""
        if not self.api_key or not self.cx:
            logger.warning("Google API credentials not configured")
            return None
        
        # Clean and optimize query
        cleaned_query = self._clean_query(query)
        
        params = {
            'q': cleaned_query,
            'key': self.api_key,
            'cx': self.cx,
            'num': min(num_results, 10),  # Google API limit
            'safe': 'medium',
            'lr': 'lang_en'  # English results
            # Note: 'sort' parameter removed as it's not supported by all CSEs
        }
        
        # Add search type specific parameters
        if search_type == "news":
            params['tbm'] = 'nws'
        elif search_type == "images":
            params['tbm'] = 'isch'
        elif search_type == "academic":
            params['as_sitesearch'] = 'scholar.google.com'
        
        try:
            logger.info(f"Performing Google search for: {cleaned_query}")
            response = self.session.get(self.base_url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            items_count = len(data.get('items', []))
            logger.info(f"Google search returned {items_count} results")
            
            # Log if there are items but empty
            if items_count == 0:
                logger.warning(f"No items in response. Response keys: {list(data.keys())}")
                if 'error' in data:
                    logger.error(f"Google API error: {data['error']}")
                else:
                    # Log search information to understand why no items
                    search_info = data.get('searchInformation', {})
                    logger.warning(f"Search info: totalResults={search_info.get('totalResults', 0)}, searchTime={search_info.get('searchTime', 0)}")
                    logger.warning(f"Full response (first 500 chars): {str(data)[:500]}")
            
            return data
            
        except requests.exceptions.Timeout:
            logger.error("Google search API timeout")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Google search API error: {e}")
            if hasattr(e.response, 'text'):
                logger.error(f"Response text: {e.response.text[:500]}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in Google search: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def _clean_query(self, query: str) -> str:
        """Clean and optimize search query"""
        # Remove special characters that might interfere with search
        cleaned = re.sub(r'[^\w\s\-\.]', ' ', query)
        
        # Remove extra whitespace
        cleaned = ' '.join(cleaned.split())
        
        # Add quotes for exact phrases if query contains specific terms
        if any(word in cleaned.lower() for word in ['what is', 'how to', 'define', 'meaning of']):
            # For definition-type queries, add quotes around key terms
            words = cleaned.split()
            if len(words) > 2:
                # Quote the main subject
                main_subject = words[-1] if words[-1].lower() not in ['is', 'to', 'of'] else words[-2]
                cleaned = cleaned.replace(main_subject, f'"{main_subject}"')
        
        return cleaned.strip()
    
    def process_results(self, search_results: Dict) -> List[Dict]:
        """Extract and structure search results with enhanced content processing"""
        if not search_results or 'items' not in search_results:
            logger.info("No Google search results found")
            return []
        
        processed = []
        for i, item in enumerate(search_results['items']):
            result = {
                'rank': i + 1,
                'title': item.get('title', ''),
                'link': item.get('link', ''),
                'snippet': item.get('snippet', ''),
                'display_link': item.get('displayLink', ''),
                'formatted_url': item.get('formattedUrl', ''),
                'pagemap': item.get('pagemap', {}),
                'highlights': [],
                'relevance_score': self._calculate_relevance(item),
                'content_type': self._detect_content_type(item)
            }
            
            # Extract and clean snippet
            snippet = item.get('snippet', '')
            if snippet:
                result['snippet'] = self._clean_snippet(snippet)
                result['highlights'] = self._extract_highlights(snippet)
            
            # Extract additional metadata
            if 'pagemap' in item:
                result['metadata'] = self._extract_metadata(item['pagemap'])
            
            processed.append(result)
        
        # Sort by relevance score
        processed.sort(key=lambda x: x['relevance_score'], reverse=True)
        return processed
    
    def _clean_snippet(self, snippet: str) -> str:
        """Clean and format snippet text"""
        # Remove HTML entities
        snippet = snippet.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
        
        # Remove extra whitespace and newlines
        snippet = ' '.join(snippet.split())
        
        # Ensure proper sentence ending
        if snippet and not snippet.endswith(('.', '!', '?')):
            snippet += '.'
        
        return snippet
    
    def _extract_highlights(self, snippet: str) -> List[str]:
        """Extract key highlights from snippet"""
        highlights = []
        
        # Split by common separators
        parts = re.split(r'[\.\-\|]', snippet)
        
        for part in parts:
            part = part.strip()
            if len(part) > 15 and len(part) < 100:  # Meaningful length
                highlights.append(part)
        
        return highlights[:3]  # Top 3 highlights
    
    def _calculate_relevance(self, item: Dict) -> float:
        """Calculate relevance score for search result"""
        score = 0.5  # Base score
        
        # Boost score for certain domains
        link = item.get('link', '').lower()
        if any(domain in link for domain in ['wikipedia.org', 'stackoverflow.com', 'github.com']):
            score += 0.3
        
        # Boost score for longer, more detailed snippets
        snippet = item.get('snippet', '')
        if len(snippet) > 100:
            score += 0.2
        
        # Boost score for results with structured data
        if 'pagemap' in item and item['pagemap']:
            score += 0.1
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _detect_content_type(self, item: Dict) -> str:
        """Detect the type of content based on URL and metadata"""
        link = item.get('link', '').lower()
        
        if 'wikipedia.org' in link:
            return 'encyclopedia'
        elif 'youtube.com' in link or 'youtu.be' in link:
            return 'video'
        elif 'github.com' in link:
            return 'code'
        elif 'stackoverflow.com' in link or 'stackexchange.com' in link:
            return 'qa'
        elif any(news in link for news in ['cnn.com', 'bbc.com', 'reuters.com', 'nytimes.com']):
            return 'news'
        elif any(academic in link for academic in ['scholar.google.com', 'arxiv.org', 'researchgate.net']):
            return 'academic'
        else:
            return 'web'
    
    def _extract_metadata(self, pagemap: Dict) -> Dict:
        """Extract useful metadata from pagemap"""
        metadata = {}
        
        # Extract publication date if available
        if 'metatags' in pagemap:
            for meta in pagemap['metatags']:
                if 'article:published_time' in meta:
                    metadata['published_date'] = meta['article:published_time']
                elif 'pubdate' in meta:
                    metadata['published_date'] = meta['pubdate']
        
        # Extract author if available
        if 'metatags' in pagemap:
            for meta in pagemap['metatags']:
                if 'article:author' in meta:
                    metadata['author'] = meta['article:author']
                elif 'author' in meta:
                    metadata['author'] = meta['author']
        
        return metadata
    
    def generate_summary(self, results: List[Dict]) -> Dict:
        """Generate comprehensive summary from search results"""
        if not results:
            return {
                'total_results': 0,
                'summary': 'No search results available',
                'key_points': [],
                'sources_by_type': {},
                'top_domains': []
            }
        
        # Extract and process content
        snippets = [result['snippet'] for result in results if result['snippet']]
        titles = [result['title'] for result in results if result['title']]
        
        # Generate comprehensive summary
        summary_text = self._create_comprehensive_summary(snippets, titles)
        
        # Extract key points with better processing
        key_points = self._extract_key_points(snippets)
        
        # Analyze sources by content type
        sources_by_type = self._categorize_sources(results)
        
        # Get top domains
        top_domains = self._get_top_domains(results)
        
        return {
            'total_results': len(results),
            'summary': summary_text,
            'key_points': key_points,
            'sources_by_type': sources_by_type,
            'top_domains': top_domains,
            'search_quality': self._assess_search_quality(results)
        }
    
    def _create_comprehensive_summary(self, snippets: List[str], titles: List[str]) -> str:
        """Create a comprehensive summary from snippets and titles"""
        if not snippets:
            return "No detailed information available from search results."
        
        # Combine snippets intelligently
        combined_text = ' '.join(snippets)
        
        # Clean and structure the text
        sentences = []
        for snippet in snippets:
            if snippet.strip():
                # Split into sentences
                snippet_sentences = re.split(r'[.!?]+', snippet)
                for sentence in snippet_sentences:
                    sentence = sentence.strip()
                    if len(sentence) > 20 and len(sentence) < 200:
                        sentences.append(sentence)
        
        # Remove duplicates and sort by length (longer = more informative)
        unique_sentences = list(set(sentences))
        unique_sentences.sort(key=len, reverse=True)
        
        # Take the most informative sentences
        summary_sentences = unique_sentences[:4]
        
        if summary_sentences:
            summary_text = '. '.join(summary_sentences)
            if not summary_text.endswith('.'):
                summary_text += '.'
        else:
            summary_text = combined_text[:500] + '...' if len(combined_text) > 500 else combined_text
        
        return summary_text
    
    def _extract_key_points(self, snippets: List[str]) -> List[str]:
        """Extract key points from snippets with improved processing"""
        key_points = set()
        
        for snippet in snippets:
            if snippet.strip():
                # Split by common separators
                parts = re.split(r'[.!?\-]', snippet)
                
                for part in parts:
                    part = part.strip()
                    # Look for meaningful phrases
                    if 20 <= len(part) <= 150 and not part.lower().startswith(('the ', 'a ', 'an ')):
                        # Clean up the phrase
                        clean_part = re.sub(r'[^\w\s\-]', '', part)
                        if len(clean_part.split()) >= 3:  # At least 3 words
                            key_points.add(clean_part)
        
        # Return top 5 most informative points
        return list(key_points)[:5]
    
    def _categorize_sources(self, results: List[Dict]) -> Dict[str, int]:
        """Categorize sources by content type"""
        categories = {}
        
        for result in results:
            content_type = result.get('content_type', 'web')
            categories[content_type] = categories.get(content_type, 0) + 1
        
        return categories
    
    def _get_top_domains(self, results: List[Dict]) -> List[str]:
        """Get top domains from search results"""
        domains = {}
        
        for result in results:
            link = result.get('link', '')
            if link:
                try:
                    from urllib.parse import urlparse
                    domain = urlparse(link).netloc
                    if domain:
                        domains[domain] = domains.get(domain, 0) + 1
                except:
                    continue
        
        # Return top 3 domains
        sorted_domains = sorted(domains.items(), key=lambda x: x[1], reverse=True)
        return [domain for domain, count in sorted_domains[:3]]
    
    def _assess_search_quality(self, results: List[Dict]) -> str:
        """Assess the quality of search results"""
        if not results:
            return "poor"
        
        # Check for high-quality sources
        quality_indicators = 0
        for result in results:
            content_type = result.get('content_type', 'web')
            if content_type in ['encyclopedia', 'academic', 'qa']:
                quality_indicators += 1
        
        quality_ratio = quality_indicators / len(results)
        
        if quality_ratio >= 0.6:
            return "excellent"
        elif quality_ratio >= 0.4:
            return "good"
        elif quality_ratio >= 0.2:
            return "fair"
        else:
            return "poor"
    
    def search_and_process(self, query: str, num_results: int = 5, search_type: str = "web") -> Dict:
        """Complete search workflow: search, process, and summarize"""
        try:
            # Perform search
            search_results = self.search(query, num_results, search_type)
            if not search_results:
                return {
                    'success': False,
                    'error': 'Search failed or no results',
                    'results': [],
                    'summary': {'total_results': 0, 'summary': '', 'key_points': []}
                }
            
            # Process results
            processed_results = self.process_results(search_results)
            
            # Generate summary
            summary = self.generate_summary(processed_results)
            
            return {
                'success': True,
                'results': processed_results,
                'summary': summary,
                'query': query,
                'search_type': search_type,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in search workflow: {e}")
            return {
                'success': False,
                'error': str(e),
                'results': [],
                'summary': {'total_results': 0, 'summary': '', 'key_points': []}
            }
    
    def multi_search(self, query: str, search_types: List[str] = None) -> Dict:
        """Perform multiple types of searches for comprehensive results"""
        if search_types is None:
            search_types = ['web', 'news', 'academic']
        
        all_results = {}
        
        for search_type in search_types:
            try:
                result = self.search_and_process(query, num_results=3, search_type=search_type)
                all_results[search_type] = result
            except Exception as e:
                logger.error(f"Error in {search_type} search: {e}")
                all_results[search_type] = {
                    'success': False,
                    'error': str(e),
                    'results': []
                }
        
        # Combine results intelligently
        combined_summary = self._combine_multi_search_results(all_results)
        
        return {
            'success': True,
            'query': query,
            'search_types': search_types,
            'results_by_type': all_results,
            'combined_summary': combined_summary,
            'timestamp': datetime.now().isoformat()
        }
    
    def _combine_multi_search_results(self, all_results: Dict) -> Dict:
        """Combine results from multiple search types"""
        combined_results = []
        total_results = 0
        
        for search_type, result in all_results.items():
            if result.get('success') and result.get('results'):
                combined_results.extend(result['results'])
                total_results += len(result['results'])
        
        if combined_results:
            # Remove duplicates based on URL
            unique_results = {}
            for result in combined_results:
                url = result.get('link', '')
                if url and url not in unique_results:
                    unique_results[url] = result
            
            combined_results = list(unique_results.values())
            
            # Generate combined summary
            summary = self.generate_summary(combined_results)
            
            return {
                'total_results': len(combined_results),
                'summary': summary,
                'search_types_used': list(all_results.keys()),
                'quality_score': self._calculate_combined_quality(all_results)
            }
        else:
            return {
                'total_results': 0,
                'summary': 'No results found across all search types',
                'search_types_used': [],
                'quality_score': 0
            }
    
    def _calculate_combined_quality(self, all_results: Dict) -> float:
        """Calculate overall quality score for combined results"""
        total_score = 0
        successful_searches = 0
        
        for search_type, result in all_results.items():
            if result.get('success'):
                successful_searches += 1
                summary = result.get('summary', {})
                quality = summary.get('search_quality', 'poor')
                
                quality_scores = {'excellent': 1.0, 'good': 0.8, 'fair': 0.6, 'poor': 0.4}
                total_score += quality_scores.get(quality, 0.4)
        
        return total_score / successful_searches if successful_searches > 0 else 0

# Global instance
google_search_service = GoogleSearchService()

