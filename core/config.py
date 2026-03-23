"""
core/config.py

Tekil yapılandırma noktası:
- Studio kök dizini (repo içi)
- OpenVINO_LLM çalışma dizini (varsayılan: C:\\OpenVINO_LLM, env ile override)
- logs/db/cache/gguf/llama-server yolları
"""

from __future__ import annotations

import os
from pathlib import Path


# Studio root: .../openvino_llm_studio (package root)
STUDIO_ROOT: Path = Path(__file__).resolve().parents[1]


def _env_path(name: str, default: str) -> Path:
    val = (os.getenv(name) or "").strip()
    return Path(val) if val else Path(default)


# User workspace root (models, gguf, llama-server, cache)
OPENVINO_LLM_HOME: Path = _env_path("OPENVINO_LLM_HOME", r"C:\OpenVINO_LLM")

# Common subpaths
LOG_DIR: Path = STUDIO_ROOT / "logs"
DB_PATH: Path = LOG_DIR / "studio.db"

OPENVINO_MODELS_DIR: Path = OPENVINO_LLM_HOME
GGUF_DIR: Path = OPENVINO_LLM_HOME / "gguf"
CACHE_DIR: Path = OPENVINO_LLM_HOME / ".cache"

LLAMA_SERVER_DIR: Path = OPENVINO_LLM_HOME / "llama-server"
LLAMA_SERVER_EXE: Path = LLAMA_SERVER_DIR / "llama-server.exe"

