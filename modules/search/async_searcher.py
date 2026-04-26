"""
modules/search/async_searcher.py

Asynchronous parallel web search implementation.
Replaces sequential search with concurrent execution for multiple queries.

Features:
- Parallel search execution with asyncio.gather
- Rate limiting to avoid API throttling
- Result deduplication and merging
- Timeout handling per query
"""

import asyncio
import logging
import time
from typing import Optional
from dataclasses import dataclass
from collections import defaultdict

from core.constants import SearchConfig

logger = logging.getLogger(__name__)


@dataclass
class SearchQuery:
    """Search query with metadata."""
    query: str
    priority: int = 0
    timeout: float = 30.0


@dataclass
class SearchResult:
    """Search result with scoring."""
    title: str
    url: str
    snippet: str
    relevance_score: float = 0.0
    rank: int = 0
    source_query: str = ""


class RateLimiter:
    """
    Token bucket rate limiter for API calls.
    
    Prevents throttling by limiting requests per time window.
    """
    
    def __init__(self, rate: int, window_seconds: float):
        self.rate = rate
        self.window = window_seconds
        self.tokens = rate
        self.last_update = time.time()
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire a token, waiting if necessary."""
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_update
            
            # Replenish tokens
            self.tokens = min(
                self.rate,
                self.tokens + elapsed * (self.rate / self.window)
            )
            self.last_update = now
            
            if self.tokens < 1:
                wait_time = (1 - self.tokens) * (self.window / self.rate)
                logger.debug(f"Rate limit: waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1


class AsyncWebSearcher:
    """
    Asynchronous web search with parallel execution.
    
    Usage:
        searcher = AsyncWebSearcher(ddgs_client)
        results = await searcher.search_multiple(queries, num_results=5)
    """
    
    def __init__(self, ddgs_client, db_manager=None, ranker=None):
        self._ddgs = ddgs_client
        self.db = db_manager
        self.ranker = ranker
        self._rate_limiter = RateLimiter(
            rate=SearchConfig.SEARCH_RATE_LIMIT,
            window_seconds=SearchConfig.SEARCH_RATE_LIMIT_WINDOW
        )
    
    async def search_single(
        self,
        query: str,
        num_results: int = 5,
        region: str = "tr-tr",
        timeout: float = 30.0
    ) -> list[SearchResult]:
        """
        Execute single search query with timeout.
        
        Args:
            query: Search query string
            num_results: Number of results to return
            region: Geographic region for search
            timeout: Request timeout in seconds
        
        Returns:
            List of SearchResult objects
        """
        try:
            # Apply rate limiting
            await self._rate_limiter.acquire()
            
            # Execute search with timeout
            loop = asyncio.get_event_loop()
            raw_results = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: list(self._ddgs.text(
                        query,
                        region=region,
                        safesearch="moderate",
                        max_results=num_results * SearchConfig.SEARCH_CANDIDATE_MULTIPLIER
                    ))
                ),
                timeout=timeout
            )
            
            # Convert to SearchResult objects
            results = [
                SearchResult(
                    title=r.get("title", ""),
                    url=r.get("href", r.get("link", "")),
                    snippet=r.get("body", r.get("snippet", "")),
                    source_query=query,
                )
                for r in raw_results
            ]
            
            logger.info(f"Search '{query}': {len(results)} results")
            return results
            
        except asyncio.TimeoutError:
            logger.warning(f"Search timeout for query: {query}")
            return []
        except Exception as e:
            logger.error(f"Search error for '{query}': {e}")
            return []
    
    async def search_multiple(
        self,
        queries: list[str],
        num_results: int = 5,
        region: str = "tr-tr",
        timeout_per_query: float = 30.0
    ) -> list[SearchResult]:
        """
        Execute multiple searches in parallel.
        
        Args:
            queries: List of search queries
            num_results: Results per query
            region: Geographic region
            timeout_per_query: Timeout per query
        
        Returns:
            Merged and deduplicated list of SearchResult
        """
        if not queries:
            return []
        
        # Execute all searches in parallel
        tasks = [
            self.search_single(
                query=q,
                num_results=num_results,
                region=region,
                timeout=timeout_per_query
            )
            for q in queries
        ]
        
        results_all = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Merge results
        merged = []
        for result in results_all:
            if isinstance(result, list):
                merged.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Search task failed: {result}")
        
        # Deduplicate by URL
        seen_urls = set()
        deduped = []
        for r in merged:
            if r.url not in seen_urls:
                seen_urls.add(r.url)
                deduped.append(r)
        
        logger.info(
            f"Parallel search: {len(queries)} queries → "
            f"{len(deduped)} unique results"
        )
        
        return deduped
    
    async def search_with_ranking(
        self,
        original_query: str,
        optimized_queries: list[str],
        num_results: int = 5,
        region: str = "tr-tr"
    ) -> list[SearchResult]:
        """
        Search with hybrid ranking (BM25 + semantic).
        
        Args:
            original_query: Original user query (for scoring)
            optimized_queries: LLM-optimized search queries
            num_results: Final number of results
            region: Geographic region
        
        Returns:
            Ranked and scored SearchResult list
        """
        # Parallel search
        all_results = await self.search_multiple(
            queries=optimized_queries,
            num_results=num_results,
            region=region
        )
        
        if not all_results:
            return []
        
        # Apply ranking if ranker available
        if self.ranker:
            ranked = self.ranker.rank(original_query, [
                {
                    "title": r.title,
                    "body": r.snippet,
                    "href": r.url,
                }
                for r in all_results
            ])
            
            # Convert back to SearchResult with scores
            final_results = []
            for r in ranked[:int(num_results * SearchConfig.MAX_CONTEXT_RESULTS_MULTIPLIER)]:
                final_results.append(SearchResult(
                    title=r.title,
                    url=r.url,
                    snippet=r.snippet,
                    relevance_score=r.relevance_score,
                    rank=r.rank,
                ))
            
            return final_results
        
        # No ranker: return as-is with basic scoring
        for i, r in enumerate(all_results[:num_results]):
            r.relevance_score = 1.0 / (i + 1)
            r.rank = i + 1
        
        return all_results
    
    def format_context(
        self,
        results: list[SearchResult],
        max_chars: int = SearchConfig.MAX_CONTEXT_CHARS
    ) -> str:
        """Format search results as context for LLM."""
        if not results:
            return ""
        
        parts = ["=== Web Arama Sonuçları ===\n"]
        total = 0
        
        # Sort by relevance
        sorted_results = sorted(
            results,
            key=lambda r: r.relevance_score,
            reverse=True
        )
        
        for r in sorted_results:
            entry = (
                f"[{r.rank}] {r.title}\n"
                f"URL: {r.url}\n"
                f"Özet: {r.snippet}\n"
                f"---\n"
            )
            
            if total + len(entry) > max_chars:
                break
            
            parts.append(entry)
            total += len(entry)
        
        return "".join(parts)
