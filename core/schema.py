from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime


class InferenceParams(BaseModel):
    """LLM üretim parametrelerini doğrulayan şema."""
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=1)
    repetition_penalty: float = Field(default=1.1, ge=1.0)


class SearchResultSchema(BaseModel):
    """Arama motoru sonuçlarını doğrulayan şema."""
    title: str
    url: str
    snippet: str
    relevance_score: float = Field(default=0.0)
    rank: int = Field(default=0)


class EnrichmentResultSchema(BaseModel):
    """DSPy zenginleştirme sonuç şeması."""
    original_prompt: str
    enriched_prompt: str
    mode: str
    mode_reason: str
    steps: List[Dict[str, Any]]
    duration_ms: float


# ═══════════════════════════════════════════════════════════════
# UNIFIED SCHEMAS
# ═══════════════════════════════════════════════════════════════

class ModelInfoSchema(BaseModel):
    """Unified model information schema."""
    name: str
    path: str
    model_type: Literal["text", "vision", "moe", "code", "embedding"]
    architecture: str
    has_tokenizer: bool
    has_config: bool
    size_mb: float = 0.0
    context_length: int = 4096
    description: str = ""


class BackendStatusSchema(BaseModel):
    """Unified backend status schema."""
    name: str
    available: bool
    loaded: bool
    model_name: Optional[str] = None
    device: str = "CPU"
    memory_usage_gb: Optional[float] = None
    error: Optional[str] = None


class ClassificationResultSchema(BaseModel):
    """DSPy classification result schema."""
    mode: Literal[
        "ChainOfThought",
        "ReAct",
        "ProgramOfThought",
        "MultiChainComparison",
        "Summarize",
        "Predict"
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    method: Literal["heuristic", "llm", "embedding", "fallback"]
    duration_ms: float


class SecurityCheckResultSchema(BaseModel):
    """Security validation result schema."""
    is_safe: bool
    risk_level: Literal["low", "medium", "high", "critical"]
    issues: List[str]
    sanitized_prompt: str
    confidence: float


class TokenUsageSchema(BaseModel):
    """Token usage statistics schema."""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    tokens_per_second: float


class MetricsSchema(BaseModel):
    """Inference metrics schema."""
    duration_ms: float
    input_tokens: int
    output_tokens: int
    tokens_per_second: float
    model_name: str
    backend: str
    device: str


class ErrorResponseSchema(BaseModel):
    """Unified error response schema."""
    success: bool = False
    error_type: str
    user_message: str
    technical_message: str
    recovery_action: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PipelineResponseSchema(BaseModel):
    """Unified pipeline response schema."""
    success: bool
    response: str
    metrics: Optional[MetricsSchema] = None
    classification: Optional[ClassificationResultSchema] = None
    search_results: Optional[List[SearchResultSchema]] = None
    error: Optional[ErrorResponseSchema] = None
