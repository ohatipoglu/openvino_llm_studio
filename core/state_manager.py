"""
core/state_manager.py
UI-bazlı state yönetimi. Her UI kendi backend, session ve cache durumunu tutar.

Bu modül, Orchestrator'un stateful problemlerini çözer:
- Her UI kendi backend seçimini yapar (çakışma yok)
- Her UI kendi session_id'sini yönetir
- Her UI kendi model cache'ini tutar
"""

import threading
import time
from typing import Dict, Optional, Any
from dataclasses import dataclass, field


@dataclass
class UIState:
    """Tek bir UI'nin state'i."""
    backend: str = "openvino"  # 'openvino', 'ollama', 'ipex'
    session_id: str = ""
    model_cache: list = field(default_factory=list)
    last_activity: float = field(default_factory=time.time)

    def __post_init__(self):
        if not self.session_id:
            self.session_id = f"sess_{int(time.time())}_{id(self)}"


class StateManager:
    """
    Thread-safe state yöneticisi.
    Her UI için ayrı state tutar (UI ID ile).
    """

    def __init__(self):
        self._states: Dict[str, UIState] = {}
        self._lock = threading.RLock()

    def get_state(self, ui_id: str) -> UIState:
        """UI için state al (yoksa oluştur)."""
        with self._lock:
            if ui_id not in self._states:
                self._states[ui_id] = UIState()
            state = self._states[ui_id]
            state.last_activity = time.time()
            return state

    def set_backend(self, ui_id: str, backend: str) -> None:
        """UI'nin backend'ini ayarla."""
        with self._lock:
            state = self.get_state(ui_id)
            if backend in ("openvino", "ollama", "ipex"):
                state.backend = backend
                state.last_activity = time.time()

    def get_backend(self, ui_id: str) -> str:
        """UI'nin backend'ini al."""
        with self._lock:
            state = self.get_state(ui_id)
            return state.backend

    def new_session(self, ui_id: str) -> str:
        """UI için yeni session başlat."""
        with self._lock:
            state = self.get_state(ui_id)
            state.session_id = f"sess_{int(time.time())}_{ui_id}"
            state.last_activity = time.time()
            return state.session_id

    def get_session_id(self, ui_id: str) -> str:
        """UI'nin session_id'sini al."""
        with self._lock:
            state = self.get_state(ui_id)
            return state.session_id

    def set_model_cache(self, ui_id: str, models: list) -> None:
        """UI'nin model cache'ini ayarla."""
        with self._lock:
            state = self.get_state(ui_id)
            state.model_cache = models
            state.last_activity = time.time()

    def get_model_cache(self, ui_id: str) -> list:
        """UI'nin model cache'ini al."""
        with self._lock:
            state = self.get_state(ui_id)
            return state.model_cache

    def clear_state(self, ui_id: str) -> None:
        """UI'nin state'ini temizle."""
        with self._lock:
            if ui_id in self._states:
                del self._states[ui_id]

    def cleanup_inactive(self, max_age_seconds: int = 3600) -> int:
        """Belirtilen süreden daha eski state'leri temizle."""
        with self._lock:
            now = time.time()
            to_remove = [
                ui_id for ui_id, state in self._states.items()
                if now - state.last_activity > max_age_seconds
            ]
            for ui_id in to_remove:
                del self._states[ui_id]
            return len(to_remove)


# Global state manager instance
_state_manager = None
_state_lock = threading.Lock()


def get_state_manager() -> StateManager:
    """Global state manager instance al (singleton)."""
    global _state_manager
    with _state_lock:
        if _state_manager is None:
            _state_manager = StateManager()
        return _state_manager
