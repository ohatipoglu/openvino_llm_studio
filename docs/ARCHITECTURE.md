# OpenVINO LLM Studio - Architecture Documentation

## 📋 Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Component Details](#component-details)
4. [Data Flow](#data-flow)
5. [Security Architecture](#security-architecture)
6. [Performance Considerations](#performance-considerations)

---

## System Overview

OpenVINO LLM Studio is a local LLM inference platform optimized for Intel Arc GPUs. It supports multiple backends and provides advanced features like web search, prompt optimization, and autonomous agents.

### Key Features

- **Multi-Backend Support**: OpenVINO, Ollama (IPEX fork), llama-cpp-python (GGUF/SYCL)
- **DSPy Prompt Optimization**: Automatic mode selection and template application
- **Web Search Integration**: DuckDuckGo + BM25 + Semantic ranking
- **ReAct Agent**: Autonomous tool calling for banking operations
- **Comprehensive Logging**: SQLite-based audit trail

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Gradio UI (ui/app.py)                       │
│  ┌────────────┬──────────────┬─────────────┬─────────────────────┐ │
│  │ Chat Tab   │ Model Gallery│ Logs Tab    │ System Stats Tab    │ │
│  └────────────┴──────────────┴─────────────┴─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Orchestrator (core/orchestrator.py)              │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Backend Manager (OpenVINO / Ollama / LlamaCpp)              │  │
│  └──────────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Pipeline Coordinator (Search → DSPy → LLM → Response)       │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  DSPy Module     │  │  Search Engine   │  │  Tool Dispatcher │
│  (dspy_enricher) │  │  (search_engine) │  │  (tools.py)      │
│ ┌──────────────┐ │  │ ┌──────────────┐ │  │ ┌──────────────┐ │
│ │ Classifier   │ │  │ │ AsyncSearch  │ │  │ │Banking Tools │ │
│ │ Templates    │ │  │ │ Ranker (BM25)│ │  │ │Query Balance │ │
│ │ Optimization │ │  │ │ Semantic     │ │  │ │Pay Bill      │ │
│ └──────────────┘ │  │ └──────────────┘ │  │ └──────────────┘ │
└──────────────────┘  └──────────────────┘  └──────────────────┘
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Backend Layer (modules/ipex_backend.py)          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ OpenVINO     │  │ Ollama       │  │ LlamaCpp     │              │
│  │ ModelLoader  │  │ REST API     │  │ (GGUF/SYCL)  │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Hardware Layer                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ Intel Arc    │  │ CPU (AVX2)   │  │ RAM (16GB+)  │              │
│  │ GPU (S