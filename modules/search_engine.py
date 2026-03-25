"""
modules/search_engine.py
DuckDuckGo web araması + akıllı sonuç sıralama.
BM25 + semantik benzerlik hibrid algoritması.
"""

import time
import logging
import re
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    relevance_score: float = 0.0
    rank: int = 0


class SearchQueryOptimizer:
    """Prompttan web araması için kısa ve etkili sorgu çıkarır."""

    # Arama için gereksiz kelimeler (Genişletilmiş liste)
    STOPWORDS_TR = {
        # Zaman zarfları
        "günümüzde", "günümüz", "bugün", "şimdi", "artık", "hâlâ", "hala",
        "geçmişte", "gelecekte", "eskiden", "yakında", "son", "yeni", "eski", "önce", "sonra",
        # Soru ekleri ve fiil kalıpları
        "bana", "bence", "acaba", "lütfen", "merhaba", "selam", "bir",
        "bu", "şu", "o", "ve", "ile", "de", "da", "mi", "mı", "mu", "mü",
        "için", "gibi", "kadar", "ama", "fakat", "ancak", "çok", "az",
        "daha", "en", "her", "hiç", "bile", "zaten", "nasıl",
        "neden", "niye", "hangi", "ne", "kim", "nerede", "bunu", "şunu", "onunu",
        "misin", "mısın", "musun", "müsün", "misiniz", "mısınız",
        "yapar", "yapabilir", "edebilir", "eder", "ediyor", "yapıyor",
        "ver", "yap", "söyle", "anlat", "açıkla", "yaz", "göster", "yanıtla", "cevapla",
        "hakkında", "konusunda", "üzerinde", "ilgili", "dair",
        "çalışılan", "çalışan", "üzerinden", "nedir", "kimdir", "hangisidir",
        "eleştir", "eleştirir", "eleştirilsel", "eleştirel", "yorumla",
        "yorum", "düşün", "düşünüyor", "değerlendir", "incele", "kısaca", "özetle",
    }
    
    STOPWORDS_EN = {
        "please", "can", "you", "could", "tell", "me", "about", "what",
        "is", "are", "how", "to", "do", "does", "did", "will", "would",
        "should", "the", "a", "an", "and", "or", "but", "in", "on",
        "at", "for", "with", "from", "that", "this", "it", "i", "my",
        "explain", "describe", "give", "write", "show", "make",
        "think", "opinion", "view", "critique", "criticize", "analyze",
    }

    def extract_query(self, prompt: str) -> str:
        """Prompttan anlamlı arama sorgusunu çıkarır."""
        text = prompt.strip()

        # Noktalama işaretlerini temizle
        text = re.sub(r"[^\w\s]", " ", text)

        # Kelimelere böl
        words = text.split()
        all_stops = self.STOPWORDS_TR | self.STOPWORDS_EN

        # Anlamlı kelimeleri filtrele (stopword değil, 2+ karakter)
        keywords = []
        for w in words:
            wl = w.lower()
            if wl not in all_stops and len(wl) > 2:
                keywords.append(w)

        # Eğer gereksizleri atınca çok az kelime kaldıysa, kelime elemeden ilk halini (kısaltarak) kullan
        if len(keywords) < 2:
            query = " ".join(words[:5])
        else:
            # En fazla 5 anahtar kelime al
            query = " ".join(keywords[:5])

        logger.debug(f"Arama sorgusu optimizasyonu: '{prompt[:60]}' -> '{query}'")
        return query.strip()


class ResultRanker:
    """
    Hibrid sıralama: BM25 + semantik benzerlik.
    sentence-transformers yoksa sadece BM25 kullanır.
    """

    def __init__(self):
        self._semantic_model = None
        self._semantic_available = False
        self._try_load_semantic()

    def _try_load_semantic(self):
        """sentence-transformers'ı yüklemeyi dene."""
        try:
            from sentence_transformers import SentenceTransformer
            # Türkçe desteği olan çok dilli model (all-MiniLM-L6-v2 sadece İngilizce içindir)
            self._semantic_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            self._semantic_available = True
            logger.info("Semantik sıralama aktif (paraphrase-multilingual-MiniLM-L12-v2)")
        except Exception:
            logger.info("sentence-transformers (multilingual) yüklenemedi, sadece BM25 kullanılacak.")

    def rank(self, query: str, results: list[dict]) -> list[SearchResult]:
        """Sonuçları sıralar ve skor atar."""
        if not results:
            return []

        # BM25 skorları
        bm25_scores = self._bm25_score(query, results)

        # Semantik skorlar
        if self._semantic_available:
            semantic_scores = self._semantic_score(query, results)
        else:
            semantic_scores = [0.0] * len(results)

        # Hibrid skor: Semantik anlama daha fazla güven (0.7), BM25'e (0.3)
        ranked = []
        for i, r in enumerate(results):
            hybrid = 0.3 * bm25_scores[i] + 0.7 * semantic_scores[i]
            ranked.append(SearchResult(
                title=r.get("title", ""),
                url=r.get("href", r.get("link", "")),
                snippet=r.get("body", r.get("snippet", "")),
                relevance_score=round(hybrid, 4),
            ))

        # Skora göre sırala
        ranked.sort(key=lambda x: x.relevance_score, reverse=True)
        for i, r in enumerate(ranked):
            r.rank = i + 1

        return ranked

    def _bm25_score(self, query: str, results: list[dict]) -> list[float]:
        try:
            from rank_bm25 import BM25Okapi
            corpus = [
                (r.get("title", "") + " " + r.get("body", r.get("snippet", ""))).lower().split()
                for r in results
            ]
            bm25 = BM25Okapi(corpus)
            scores = bm25.get_scores(query.lower().split())
            # Normalize
            max_s = max(scores) if max(scores) > 0 else 1
            return [float(s / max_s) for s in scores]
        except Exception as e:
            logger.warning(f"BM25 hatası: {e}")
            return [1.0 / (i + 1) for i in range(len(results))]

    def _semantic_score(self, query: str, results: list[dict]) -> list[float]:
        try:
            import numpy as np
            texts = [
                (r.get("title", "") + " " + r.get("body", r.get("snippet", "")))[:1024]
                for r in results
            ]
            # show_progress_bar=False eklenmedi çünkü bazı versiyonlarda desteklenmiyor olabilir, varsayılan sessizdir.
            embeddings = self._semantic_model.encode([query] + texts)
            q_emb = embeddings[0]
            doc_embs = embeddings[1:]
            # Cosine similarity
            scores = []
            for d in doc_embs:
                norm_q = np.linalg.norm(q_emb)
                norm_d = np.linalg.norm(d)
                norm = norm_q * norm_d
                score = float(np.dot(q_emb, d) / norm) if norm > 0 else 0.0
                scores.append(max(0.0, score))
            return scores
        except Exception as e:
            logger.warning(f"Semantik skor hatası: {e}")
            return [0.0] * len(results)


class WebSearcher:
    """DuckDuckGo araması yapar."""

    def __init__(self, db_manager=None):
        self.db = db_manager
        self.optimizer = SearchQueryOptimizer()
        self.ranker = ResultRanker()

    def search(self, prompt: str, num_results: int = 5,
               region: str = "tr-tr", session_id: str = "", optimize_query: bool = True) -> tuple[list[SearchResult], str]:
        """
        Ana arama fonksiyonu.
        Returns: (ranked_results, search_query)
        """
        start = time.time()
        
        if optimize_query:
            search_query = self.optimizer.extract_query(prompt)
        else:
            search_query = prompt

        raw_results = []

        try:
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                raw = list(ddgs.text(
                    search_query,
                    region=region,
                    safesearch="moderate",
                    max_results=max(10, num_results * 2)  # Sıralama için yeterli aday havuzu
                ))
                raw_results = raw
                logger.info(f"DuckDuckGo: {len(raw_results)} sonuç ({search_query})")

        except Exception as e:
            logger.error(f"DuckDuckGo arama hatası: {e}")
            if self.db:
                self.db.log_error(session_id, "WebSearcher", e,
                                  {"query": search_query, "prompt": prompt})
            return [], search_query

        # Sırala
        ranked = self.ranker.rank(search_query, raw_results)
        # İstenen sayıya kırp
        ranked = ranked[:num_results]

        duration_ms = (time.time() - start) * 1000

        # Logla
        if self.db:
            self.db.log_search(
                session_id=session_id,
                original_prompt=prompt,
                search_query=search_query,
                raw_results=[{
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", "")
                } for r in raw_results],
                ranked_results=[{
                    "rank": r.rank,
                    "title": r.title,
                    "url": r.url,
                    "snippet": r.snippet
                } for r in ranked],
                relevance_scores=[r.relevance_score for r in ranked],
                duration_ms=duration_ms
            )

        return ranked, search_query

    def format_context(self, results: list[SearchResult], max_chars: int = 15000) -> str:
        """Arama sonuçlarını LLM için bağlam metnine çevirir."""
        if not results:
            return ""

        parts = ["=== Web Arama Sonuçları ===\n"]
        total = 0
        for r in results:
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
