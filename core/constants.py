"""
core/constants.py
Centralized constants for the application.
"""

class DSPyConfig:
    CLASSIFY_TIMEOUT_S = 15
    CLASSIFY_COOLDOWN_S = 30
    SEARCH_QUERY_TIMEOUT_S = 10
    CLASSIFY_MAX_TOKENS = 12
    MAX_SEARCH_QUERIES = 3
    CONFIDENCE_THRESHOLD_HIGH = 0.8
    CONFIDENCE_THRESHOLD_LOW = 0.5
    
    MODES = {
        "ChainOfThought": {"description": "Analitik sorular", "keywords": ["acikla", "analiz"]},
        "ReAct": {"description": "Araç gerektiren", "keywords": ["transfer", "gonder"]},
        "ProgramOfThought": {"description": "Kod/math", "keywords": ["hesapla", "kod"]},
        "MultiChainComparison": {"description": "Karsilastirma", "keywords": ["karsilastir"]},
        "Summarize": {"description": "Ozet", "keywords": ["ozetle"]},
        "Predict": {"description": "Basit", "keywords": ["nedir"]},
    }

class SearchConfig:
    DEFAULT_NUM_RESULTS = 5
    BM25_WEIGHT = 0.3
    SEMANTIC_WEIGHT = 0.7
    MAX_CONTEXT_CHARS = 15000
    JUNK_DOMAINS = {"sozluk.gov.tr", "nedemek.org"}
    SEARCH_RATE_LIMIT = 10

class ModelConfig:
    MIN_AVAILABLE_MEMORY_GB = 2.0
    GPU_MAX_ALLOC_PERCENT_DEFAULT = 75
    DEFAULT_CONTEXT_LENGTH = 4096

class BackendConfig:
    LLAMA_SERVER_DEFAULT_PORT = 8080
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_MAX_TOKENS = 512
    DEFAULT_REPETITION_PENALTY = 1.1
    COMMON_STOP_TOKENS = ["