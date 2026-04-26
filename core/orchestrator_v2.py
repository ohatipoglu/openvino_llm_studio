"""
core/orchestrator_v2.py

Enhanced orchestrator with integrated security, improved DSPy classification,
async search, and unified error handling.

This is the next-generation orchestrator that incorporates all improvements:
- Security guard for prompt injection protection
- Multi-stage DSPy mode classification
- Async parallel web search
- Centralized error handling
- Hierarchical prompt templates
"""

import uuid
import logging
import threading
import asyncio
import json
import re
from typing import Optional, Generator, List
from pathlib import Path

from core.database import DatabaseManager
from core.model_manager import ModelScanner, ModelLoader, ModelInfo
from core.constants import DSPyConfig, SearchConfig, SecurityConfig
from core.prompts import BaseTemplates, DomainTemplates, ModelFormatters
from core.error_handling import (
    error_context, ErrorHandler, ErrorContext,
    ModelError, SearchError, DSPyError, SecurityError
)

# New modular imports
from modules.dspy.classifier import ModeClassifier, ClassificationResult
from modules.search.async_searcher import AsyncWebSearcher
from modules.security import PromptGuard, MessageSeparator
from modules.tools import ToolDispatcher
from modules.ipex_backend import OllamaBackend, LlamaCppBackend
from modules.ipex_worker_client import IPEXWorkerClient
from modules.hf_catalog import HFModelCatalog

logger = logging.getLogger(__name__)


class OrchestratorV2:
    """
    Next-generation orchestrator with enhanced security and performance.
    
    Features:
    - Multi-stage DSPy classification
    - Async parallel search
    - Prompt injection protection
    - Unified error handling
    - Hierarchical templates
    """
    
    BACKENDS = ("openvino", "ollama", "ipex")
    
    def __init__(self, enable_security: bool = True):
        """
        Initialize enhanced orchestrator.
        
        Args:
            enable_security: Enable prompt security validation
        """
        self.session_id = str(uuid.uuid4())[:8]
        self._lock = threading.Lock()
        self._active_backend: str = "openvino"
        self._enable_security = enable_security
        
        logger.info("OrchestratorV2 başlatılıyor...")
        
        # Core components
        self.db = DatabaseManager()
        self.scanner = ModelScanner()
        self.tool_dispatcher = ToolDispatcher()
        self.catalog = HFModelCatalog()
        
        # Security (optional)
        if self._enable_security:
            self.security_guard = PromptGuard(strict_mode=False)
            self.message_separator = MessageSeparator()
            logger.info("Security guard enabled")
        else:
            self.security_guard = None
            self.message_separator = None
        
        # Backend loaders
        self.ov_loader = ModelLoader(db_manager=self.db)
        self.ollama = OllamaBackend(db_manager=self.db)
        self.ipex = IPEXWorkerClient(
            conda_env="openvino_studio",
            port=62000,
            db_manager=self.db,
        )
        
        # Enhanced DSPy with new classifier
        self.mode_classifier = ModeClassifier(
            tool_dispatcher=self.tool_dispatcher
        )
        
        # Async search (initialized when needed)
        self._async_searcher = None
        
        # State
        self._ov_models: list = []
        self._current_model_name: str = ""
        
        # Log startup
        self.db.log_general(
            self.session_id, "INFO", "OrchestratorV2",
            "Enhanced studio başlatıldı.",
            {"session": self.session_id, "security": enable_security}
        )
        logger.info(f"OrchestratorV2 hazır. Session: {self.session_id}")
    
    @property
    def async_searcher(self):
        """Lazy initialization of async searcher."""
        if self._async_searcher is None:
            # Initialize with DDGS client from existing searcher
            try:
                from ddgs import DDGS
                ddgs_client = DDGS()
                from modules.search_engine import ResultRanker
                ranker = ResultRanker()
                self._async_searcher = AsyncWebSearcher(
                    ddgs_client=ddgs_client,
                    db_manager=self.db,
                    ranker=ranker
                )
            except Exception as e:
                logger.warning(f"Async searcher initialization failed: {e}")
                self._async_searcher = None
        return self._async_searcher
    
    # ═══════════════════════════════════════════════════════════════
    # SECURITY
    # ═══════════════════════════════════════════════════════════════
    
    def _validate_prompt(self, prompt: str) -> tuple[bool, str]:
        """
        Validate prompt for security issues.
        
        Args:
            prompt: User input prompt
        
        Returns:
            (is_safe, sanitized_prompt or error_message)
        """
        if not self._enable_security or not self.security_guard:
            return True, prompt
        
        check = self.security_guard.validate(prompt)
        
        if not check.is_safe:
            logger.warning(
                f"Security check failed: {check.risk_level} - {check.issues}"
            )
            return False, f"Security validation failed: {', '.join(check.issues)}"
        
        return True, check.sanitized_prompt
    
    def _safe_format_prompt(
        self,
        template: str,
        user_content: str,
        context: dict = None
    ) -> str:
        """
        Safely format prompt with user content.
        
        Args:
            template: Prompt template
            user_content: User content to insert
            context: Additional context
        
        Returns:
            Safe formatted prompt
        """
        if self.security_guard and self._enable_security:
            return self.security_guard.safe_format(
                template, user_content, field_name="user_input"
            )
        else:
            return template.format(user_input=user_content)
    
    # ═══════════════════════════════════════════════════════════════
    # DSPy MODE CLASSIFICATION
    # ═══════════════════════════════════════════════════════════════
    
    def classify_prompt(
        self,
        prompt: str,
        use_llm: bool = True
    ) -> ClassificationResult:
        """
        Classify prompt using multi-stage pipeline.
        
        Args:
            prompt: User input
            use_llm: Whether to use LLM for classification
        
        Returns:
            ClassificationResult with mode, confidence, reasoning
        """
        with error_context(
            "OrchestratorV2", "classify_prompt",
            params={"prompt_length": len(prompt), "use_llm": use_llm},
            state={"session_id": self.session_id}
        ):
            loader = self._active_loader
            result = self.mode_classifier.classify(
                prompt=prompt,
                use_llm=use_llm,
                loader=loader if use_llm else None
            )
            
            logger.info(
                f"Classification: {result.mode} ({result.confidence:.2f}) "
                f"via {result.method}"
            )
            
            return result
    
    # ═══════════════════════════════════════════════════════════════
    # ASYNC SEARCH
    # ═══════════════════════════════════════════════════════════════
    
    async def _search_async(
        self,
        query: str,
        num_results: int = 5,
        region: str = "tr-tr"
    ) -> list:
        """
        Execute async web search.
        
        Args:
            query: Search query
            num_results: Number of results
            region: Geographic region
        
        Returns:
            List of search results
        """
        if not self.async_searcher:
            logger.warning("Async searcher not available, falling back to sync")
            # Fallback to existing sync search
            from modules.search_engine import WebSearcher
            fallback_searcher = WebSearcher(db_manager=self.db)
            results, _ = fallback_searcher.search(
                prompt=query,
                num_results=num_results,
                region=region,
                session_id=self.session_id
            )
            return results
        
        results = await self.async_searcher.search_single(
            query=query,
            num_results=num_results,
            region=region,
            timeout=SearchConfig.SEARCH_RATE_LIMIT_WINDOW
        )
        
        return results
    
    async def _search_multiple_async(
        self,
        queries: list[str],
        num_results: int = 5,
        region: str = "tr-tr"
    ) -> list:
        """
        Execute multiple searches in parallel.
        
        Args:
            queries: List of queries
            num_results: Results per query
            region: Geographic region
        
        Returns:
            Merged and deduplicated results
        """
        if not self.async_searcher:
            # Fallback to sequential sync search
            all_results = []
            for q in queries:
                results = await self._search_async(q, num_results, region)
                all_results.extend(results)
            return all_results
        
        return await self.async_searcher.search_multiple(
            queries=queries,
            num_results=num_results,
            region=region
        )
    
    def _generate_search_queries(self, prompt: str) -> list[str]:
        """
        Generate search queries from prompt.
        
        Args:
            prompt: User prompt
        
        Returns:
            List of search queries
        """
        loader = self._active_loader
        if not (loader and loader.is_loaded):
            return [prompt[:60]]
        
        # Use DSPy enricher's query generation
        try:
            queries = self.enricher.generate_search_query(prompt)
            return queries[:DSPyConfig.MAX_SEARCH_QUERIES]
        except Exception as e:
            logger.warning(f"Query generation failed: {e}")
            return [prompt[:60]]
    
    # ═══════════════════════════════════════════════════════════════
    # PROMPT TEMPLATES
    # ═══════════════════════════════════════════════════════════════
    
    def _apply_template(
        self,
        mode: str,
        prompt: str,
        context: str = "",
        system_prompt: str = ""
    ) -> str:
        """
        Apply hierarchical template for mode.
        
        Args:
            mode: DSPy mode
            prompt: User prompt
            context: Search context
            system_prompt: System prompt
        
        Returns:
            Formatted prompt
        """
        tool_desc = self.tool_dispatcher.get_tool_descriptions()
        
        # Select template based on mode
        if mode == "ReAct":
            # Check if banking domain
            banking_keywords = ["transfer", "gönder", "öde", "bakiye", "fatura"]
            if any(kw in prompt.lower() for kw in banking_keywords):
                template = DomainTemplates.banking_react(
                    tool_desc=tool_desc,
                    context=context,
                    question=prompt
                )
            else:
                template = BaseTemplates.react(
                    tool_desc=tool_desc,
                    context=context,
                    question=prompt
                )
        elif mode == "ChainOfThought":
            template = BaseTemplates.chain_of_thought(
                context=context,
                question=prompt
            )
        elif mode == "ProgramOfThought":
            template = BaseTemplates.program_of_thought(
                context=context,
                problem=prompt
            )
        elif mode == "MultiChainComparison":
            template = BaseTemplates.multi_chain_comparison(
                context=context,
                topic=prompt
            )
        elif mode == "Summarize":
            template = BaseTemplates.summarize(
                context=context,
                topic=prompt
            )
        else:  # Predict
            template = BaseTemplates.predict(question=prompt)
        
        # Add system prompt if provided
        if system_prompt and mode != "ReAct":  # ReAct has its own system message
            template = f"System: {system_prompt}\n\n{template}"
        
        return template
    
    # ═══════════════════════════════════════════════════════════════
    # ENHANCED PIPELINE
    # ═══════════════════════════════════════════════════════════════
    
    def run_pipeline_v2(
        self,
        prompt: str,
        params: dict,
        enable_search: bool = True,
        enable_dspy: bool = True,
        num_search_results: int = 5,
        system_prompt: str = "",
        use_async_search: bool = True
    ) -> Generator[str, None, None]:
        """
        Enhanced inference pipeline with all improvements.
        
        Args:
            prompt: User input (will be security validated)
            params: Inference parameters
            enable_search: Enable web search
            enable_dspy: Enable DSPy classification
            num_search_results: Number of search results
            system_prompt: Optional system prompt
            use_async_search: Use async parallel search
        
        Yields:
            Streaming response chunks
        """
        loader = self._active_loader
        
        if not loader.is_loaded:
            yield "❌ Model yüklü değil. Lütfen önce bir model seçin ve yükleyin."
            return
        
        # Step 1: Security validation
        if self._enable_security and self.security_guard:
            is_safe, validated_prompt = self._validate_prompt(prompt)
            if not is_safe:
                yield f"⚠️ Güvenlik kontrolü başarısız: {validated_prompt}"
                return
            prompt = validated_prompt
        
        try:
            # Step 2: DSPy classification
            classification = None
            if enable_dspy:
                yield "🧠 Prompt sınıflandırılıyor (DSPy)...\n"
                classification = self.classify_prompt(prompt, use_llm=True)
                mode = classification.mode
                yield f"✅ DSPy modu: **{mode}** ({classification.reason})\n"
            else:
                mode = "ChainOfThought"  # Default
            
            # Step 3: Web search (async if available)
            search_context = ""
            if enable_search:
                yield "🔍 Web araması başlatılıyor...\n"
                
                # Generate queries
                queries = self._generate_search_queries(prompt)
                yield f"✨ {len(queries)} arama sorgusu: {', '.join(queries)}\n"
                
                try:
                    if use_async_search:
                        # Async parallel search
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        all_results = loop.run_until_complete(
                            self._search_multiple_async(
                                queries=queries,
                                num_results=num_search_results,
                                region="tr-tr"
                            )
                        )
                        loop.close()
                    else:
                        # Fallback to sync search
                        from modules.search_engine import WebSearcher
                        searcher = WebSearcher(db_manager=self.db)
                        all_results = []
                        for q in queries:
                            results, _ = searcher.search(
                                prompt=q,
                                num_results=num_search_results,
                                region="tr-tr",
                                session_id=self.session_id
                            )
                            all_results.extend(results)
                    
                    # Filter and format results
                    if all_results:
                        # Deduplicate by URL
                        unique_urls = set()
                        filtered_results = []
                        for r in all_results:
                            url = getattr(r, 'url', r.get('url', ''))
                            if url not in unique_urls:
                                # Filter junk domains
                                if not any(d in url for d in SearchConfig.JUNK_DOMAINS):
                                    unique_urls.add(url)
                                    filtered_results.append(r)
                        
                        if filtered_results:
                            # Format context
                            if hasattr(self.async_searcher, 'format_context'):
                                search_context = self.async_searcher.format_context(
                                    filtered_results
                                )
                            else:
                                from modules.search_engine import WebSearcher
                                search_context = WebSearcher.format_context(
                                    WebSearcher,
                                    filtered_results
                                )
                            yield f"✅ {len(filtered_results)} sonuç bağlama eklendi.\n"
                        else:
                            yield "⚠️ Kaliteli arama sonucu bulunamadı.\n"
                    else:
                        yield "⚠️ Arama sonucu bulunamadı.\n"
                        
                except Exception as e:
                    logger.error(f"Search error: {e}")
                    self.db.log_error(self.session_id, "Pipeline.search", e)
                    yield f"⚠️ Arama hatası: {e}\n"
            
            # Step 4: Apply template
            final_prompt = self._apply_template(
                mode=mode,
                prompt=prompt,
                context=search_context,
                system_prompt=system_prompt
            )
            
            # Step 5: Generate response
            yield "⚡ Model yanıt üretiyor...\n\n"
            
            response, metrics = loader.generate(
                prompt=final_prompt,
                params=params,
                session_id=self.session_id,
                raw_prompt=prompt,
                system_prompt=system_prompt,
            )
            
            if metrics.get("error"):
                yield f"❌ Model hatası: {metrics['error']}"
                return
            
            yield "---\n"
            yield response
            
            # Metrics
            tps = metrics.get("tokens_per_second", "?")
            dur = metrics.get("duration_ms", 0)
            tok = metrics.get("output_tokens", "?")
            dur_fmt = f"{float(dur):.0f}ms" if dur else "?"
            yield (
                f"\n\n---\n📊 *{tok} token | {tps} tok/s | "
                f"{dur_fmt} | {self._active_backend}*"
            )
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            self.db.log_error(
                self.session_id, "Pipeline.general", e,
                {"prompt": prompt[:200]}
            )
            yield f"\n❌ Beklenmeyen hata: {e}"
    
    # ═══════════════════════════════════════════════════════════════
    # BACKEND MANAGEMENT (inherited)
    # ═══════════════════════════════════════════════════════════════
    
    def set_backend(self, backend: str):
        """Switch active backend."""
        with self._lock:
            if backend in self.BACKENDS:
                self._active_backend = backend
                logger.info(f"Backend değiştirildi: {backend}")
    
    @property
    def _active_loader(self):
        """Get active backend loader."""
        if self._active_backend == "ollama":
            return self.ollama
        if self._active_backend == "ipex":
            return self.ipex
        return self.ov_loader
    
    # Additional methods from original orchestrator...
    # (scan_models, load_model, etc. can be inherited or delegated)
