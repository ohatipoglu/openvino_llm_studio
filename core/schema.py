from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

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
