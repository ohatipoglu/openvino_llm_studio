# API Reference

## Core Modules

### Orchestrator (`core/orchestrator.py`)

Main coordinator for all LLM operations.

```python
class Orchestrator:
    """
    Main coordinator for LLM inference pipeline.
    
    Responsibilities:
    - Backend management (OpenVINO/Ollama/LlamaCpp)
    - Model scanning and loading
    - Pipeline execution (search → DSPy → LLM)
    - ReAct agent coordination
    """
    
    def __init__(self) -> None:
        """Initialize orchestrator with all components."""
        
    def set_backend(self, backend: str) -> None:
        """
        Switch active backend.
        
        Args:
            backend: One of 'openvino', 'ollama', 'ipex'
        """
        
    def load_model(self, model_path: str, device: str = "CPU",
                   ov_config: dict = None) -> tuple[bool, str]:
        """
        Load model into memory.
        
        Args:
            model_path: Path to model directory or GGUF file
            device: 'CPU', 'GPU', or 'AUTO'
            ov_config: OpenVINO configuration overrides
        
        Returns:
            (success, message) tuple
        """
        
    def run_pipeline(self, prompt: str, params: dict,
                     enable_search: bool = True,
                     enable_dspy: bool = True,
                     num_search_results: int = 5,
                     system_prompt: str = "") -> Generator[str, None, None]:
        """
        Execute full inference pipeline.
        
        Args:
            prompt: User input prompt
            params: Inference parameters (temperature, max_tokens, etc.)
            enable_search: Enable web search
            enable_dspy: Enable DSPy prompt optimization
            num_search_results: Number of search results to fetch
            system_prompt: Optional system prompt
        
        Yields:
            Streaming response chunks
        """
```

---

### DSPy Components

#### Mode Classifier (`modules/dspy/classifier.py`)

```python
class ModeClassifier:
    """
    Multi-stage DSPy mode classifier.
    
    Classification pipeline:
    1. Heuristic check (fast, high-confidence)
    2. Embedding-based semantic similarity
    3. LLM classification with few-shot examples
    4. Rule-based fallback
    """
    
    def classify(self, prompt: str, use_llm: bool = True,
                 loader=None) -> ClassificationResult:
        """
        Classify prompt into optimal DSPy mode.
        
        Args:
            prompt: User input
            use_llm: Whether to use LLM for classification
            loader: Model loader for LLM calls
        
        Returns:
            ClassificationResult with mode, confidence, reasoning
        """
```

#### Prompt Templates (`core/prompts.py`)

```python
class BaseTemplates:
    """Base prompt templates for each DSPy mode."""
    
    @staticmethod
    def chain_of_thought(context: str, question: str) -> str:
        """Step-by-step reasoning template."""
        
    @staticmethod
    def react(tool_desc: str, context: str, question: str) -> str:
        """ReAct template for tool-based tasks."""
        
    @staticmethod
    def program_of_thought(context: str, problem: str) -> str:
        """Mathematical/code problem solving."""
        
    @staticmethod
    def multi_chain_comparison(context: str, topic: str) -> str:
        """Comparative analysis template."""
        
    @staticmethod
    def summarize(context: str, topic: str) -> str:
        """Summarization template."""
        
    @staticmethod
    def predict(question: str) -> str:
        """Simple factual question."""


class DomainTemplates:
    """Domain-specific template variants."""
    
    @staticmethod
    def banking_react(tool_desc: str, context: str, question: str) -> str:
        """ReAct template for banking operations."""
        
    @staticmethod
    def research_react(tool_desc: str, context: str, question: str) -> str:
        """ReAct template for research tasks."""
        
    @staticmethod
    def code_assistant(context: str, task: str) -> str:
        """Programming task template."""
```

---

### Search Engine (`modules/search/`)

```python
class AsyncWebSearcher:
    """
    Asynchronous web search with parallel execution.
    
    Features:
    - Parallel search with asyncio.gather
    - Rate limiting
    - Result deduplication
    - Hybrid ranking (BM25 + semantic)
    """
    
    async def search_multiple(
        self, queries: list[str],
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
            Merged and deduplicated SearchResult list
        """
        
    async def search_with_ranking(
        self, original_query: str,
        optimized_queries: list[str],
        num_results: int = 5
    ) -> list[SearchResult]:
        """
        Search with hybrid ranking.
        
        Args:
            original_query: Original user query (for scoring)
            optimized_queries: LLM-optimized queries
            num_results: Final number of results
        
        Returns:
            Ranked SearchResult list
        """


class ResultRanker:
    """
    Hybrid result ranking: BM25 + semantic similarity.
    
    Weights:
    - BM25: 0.3
    - Semantic: 0.7
    """
    
    def rank(self, query: str, results: list[dict]) -> list[SearchResult]:
        """
        Rank search results.
        
        Args:
            query: Search query
            results: Raw search results
        
        Returns:
            Ranked SearchResult list
        """
```

---

### Security (`modules/security/`)

```python
class PromptGuard:
    """
    Security guard for LLM prompts.
    
    Protects against:
    - Prompt injection
    - Jailbreak attempts
    - System prompt leakage
    - XSS/SQL injection
    """
    
    def validate(self, prompt: str) -> SecurityCheck:
        """
        Validate prompt for security issues.
        
        Args:
            prompt: User input
        
        Returns:
            SecurityCheck with validation results
        """
        
    def safe_format(self, template: str, user_content: str,
                    field_name: str = "user_input") -> str:
        """
        Safely insert user content into template.
        
        Args:
            template: Prompt template
            user_content: User content
            field_name: Placeholder name
        
        Returns:
            Safe formatted prompt
        """


class MessageSeparator:
    """
    Role-based message separation.
    
    Prevents injection by clearly separating:
    - System instructions
    - User messages
    - Assistant messages
    - Tool outputs
    """
    
    def format_messages(self, messages: list[dict]) -> str:
        """
        Format messages with role delimiters.
        
        Args:
            messages: List of {role, content} dicts
        
        Returns:
            Formatted prompt string
        """
```

---

## Configuration

### Constants (`core/constants.py`)

```python
class DSPyConfig:
    """DSPy configuration."""
    CLASSIFY_TIMEOUT_S = 15.0
    CLASSIFY_COOLDOWN_S = 30.0
    MAX_SEARCH_QUERIES = 3
    MODES = {...}  # Mode definitions


class SearchConfig:
    """Search engine configuration."""
    DEFAULT_NUM_RESULTS = 5
    BM25_WEIGHT = 0.3
    SEMANTIC_WEIGHT = 0.7
    JUNK_DOMAINS = {...}


class ModelConfig:
    """Model loading configuration."""
    MIN_AVAILABLE_MEMORY_GB = 2.0
    GPU_MAX_ALLOC_PERCENT_DEFAULT = 75
    KV_CACHE_PRECISION_DEFAULT = "u8"
```

---

## Data Models

```python
@dataclass
class ClassificationResult:
    """DSPy classification result."""
    mode: str
    confidence: float
    reason: str
    method: str  # 'heuristic', 'llm', 'embedding', 'fallback'
    duration_ms: float


@dataclass
class SearchResult:
    """Search result."""
    title: str
    url: str
    snippet: str
    relevance_score: float
    rank: int
    source_query: str


@dataclass
class SecurityCheck:
    """Security validation result."""
    is_safe: bool
    issues: list[str]
    sanitized_prompt: str
    risk_level: str  # 'low', 'medium', 'high', 'critical'
```
