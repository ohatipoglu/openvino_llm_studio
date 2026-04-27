"""
tests/test_app.py
Basit duman testleri - uygulamanın başlatılabilir olduğunu doğrular.
"""

import sys
from pathlib import Path

# Proje kökünü path'e ekle
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def test_import_orchestrator():
    """Orchestrator import edilebilir mi?"""
    from core.orchestrator import Orchestrator
    assert Orchestrator is not None


def test_import_state_manager():
    """StateManager import edilebilir mi?"""
    from core.state_manager import StateManager, get_state_manager
    assert StateManager is not None
    assert get_state_manager is not None


def test_state_manager_singleton():
    """StateManager singleton pattern çalışıyor mu?"""
    from core.state_manager import get_state_manager
    
    sm1 = get_state_manager()
    sm2 = get_state_manager()
    
    assert sm1 is sm2, "StateManager singleton olmalı"


def test_state_manager_backend():
    """StateManager backend ayarlama çalışıyor mu?"""
    from core.state_manager import get_state_manager
    
    sm = get_state_manager()
    sm.set_backend("test-ui-1", "openvino")
    
    assert sm.get_backend("test-ui-1") == "openvino"
    
    sm.set_backend("test-ui-1", "ollama")
    assert sm.get_backend("test-ui-1") == "ollama"


def test_state_manager_session():
    """StateManager session yönetimi çalışıyor mu?"""
    from core.state_manager import get_state_manager
    
    sm = get_state_manager()
    session_id = sm.new_session("test-ui-2")
    
    assert session_id is not None
    assert session_id.startswith("sess_")
    assert sm.get_session_id("test-ui-2") == session_id


def test_import_ui_unified():
    """Unified UI import edilebilir mi?"""
    from ui.app_unified import build_unified_ui
    assert build_unified_ui is not None


def test_import_ui_modern():
    """Modern UI import edilebilir mi?"""
    from ui.app_modern import build_modern_ui
    assert build_modern_ui is not None


def test_import_ui_workspace():
    """Workspace UI import edilebilir mi?"""
    from ui.app_workspace import build_workspace_ui
    assert build_workspace_ui is not None


def test_import_database():
    """DatabaseManager import edilebilir mi?"""
    from modules.database import DatabaseManager
    assert DatabaseManager is not None


def test_import_search_engine():
    """WebSearcher import edilebilir mi?"""
    from modules.search_engine import WebSearcher
    assert WebSearcher is not None


def test_import_dspy_enricher():
    """DSPyEnricher import edilebilir mi?"""
    from modules.dspy_enricher import DSPyEnricher
    assert DSPyEnricher is not None
