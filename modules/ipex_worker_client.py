"""
modules/ipex_worker_client.py
==============================
LlamaCppBackend'i ana uygulamadan yöneten istemci sarmalayıcı.

ipex-llm EOL Notu (Mart 2026):
  IPEX-LLM projesi kullanımdan kaldırıldı. Bu modül artık ayrı bir
  conda env + TCP socket mimarisi yerine LlamaCppBackend sınıfını
  doğrudan sarmalar. Böylece:
    • Gereksiz subprocess/TCP ek yükü ortadan kalkar.
    • llama-cpp-python aynı Python ortamında çalışır (openvino_studio env).
    • Intel Arc iGPU desteği için SYCL derlemeli llama-cpp-python
      veya IPEX Ollama fork kullanılır.

Kullanım (orchestrator.py içinden):
    client = IPEXWorkerClient()
    client.start_worker()          # no-op, geriye uyumluluk için
    client.load("C:/gguf/q4.gguf", device="xpu")
    response, metrics = client.generate(prompt, params)
    client.stop_worker()           # no-op
"""

import logging
import threading
from pathlib import Path
from typing import Optional

from modules.ipex_backend import LlamaCppBackend, LlamaServerBackend
from core.config import LLAMA_SERVER_EXE

logger = logging.getLogger(__name__)


class IPEXWorkerClient:
    """
    LlamaCppBackend üzerinde ince bir sarmalayıcı.

    Eski ipex_worker.py / TCP socket mimarisinin yerini alır;
    dışarıya aynı arayüzü sunar (start_worker, stop_worker, load,
    generate, generate_raw, get_status, get_memory_info).

    Cihaz seçimi:
        "cpu"  → n_gpu_layers=0 (tam CPU)
        "xpu"  → n_gpu_layers=-1 (SYCL/XPU — SYCL derlemesi gerekir)
        "gpu"  → n_gpu_layers=-1 (aynı)
        "auto" → n_gpu_layers=-1 (deneysel)
    """

    # llama-server varsayılan ayarları
    LLAMA_SERVER_EXE  = str(LLAMA_SERVER_EXE)
    LLAMA_SERVER_HOST = "127.0.0.1"
    LLAMA_SERVER_PORT = 8080

    def __init__(self,
                 conda_env: str = "openvino_studio",   # artık kullanılmıyor, uyumluluk için
                 port: int = 62000,                     # artık kullanılmıyor, uyumluluk için
                 db_manager=None,
                 use_llama_server: bool = True):
        self.db        = db_manager
        self._lock     = threading.Lock()
        self._device: str = "cpu"

        # Backend seçimi: llama-server (önerilen) veya llama-cpp-python
        self._use_llama_server = use_llama_server
        if use_llama_server:
            self._server_backend = LlamaServerBackend(db_manager=db_manager)
            self._backend        = LlamaCppBackend(db_manager=db_manager)  # fallback
            logger.info("IPEXWorkerClient: LlamaServerBackend (llama-server.exe) aktif.")
        else:
            self._server_backend = None
            self._backend        = LlamaCppBackend(db_manager=db_manager)
            logger.info("IPEXWorkerClient: LlamaCppBackend (llama-cpp-python) aktif.")

        # Eski parametreler yok sayılır; loglama için saklanır
        self._legacy_conda_env = conda_env
        self._legacy_port      = port

    # ─────────────────────── Worker Yaşam Döngüsü ────────────────
    # Geriye uyumluluk — artık subprocess başlatılmıyor.

    def start_worker(self) -> tuple[bool, str]:
        """
        Geriye uyumluluk için korundu.
        llama-cpp-python aynı süreçte çalıştığından ayrı bir worker gerekmez.
        llama-cpp-python kurulumunu doğrular.
        """
        if not self._backend.is_available():
            return False, (
                "llama-cpp-python kurulu değil.\n\n"
                "Kurulum seçenekleri:\n"
                "  1) CPU (kolay):\n"
                "     pip install llama-cpp-python\n\n"
                "  2) Intel Arc XPU/iGPU (SYCL — yüksek performans):\n"
                "     # Intel oneAPI Base Toolkit kurulu olmalı\n"
                "     set CMAKE_ARGS=-DGGML_SYCL=ON\n"
                "     pip install llama-cpp-python --no-binary llama-cpp-python\n\n"
                "  3) Alternatif: Ollama + IPEX fork\n"
                "     https://github.com/ipex-llm/ollama/releases"
            )
        logger.info("IPEXWorkerClient: LlamaCppBackend hazır (aynı süreç).")
        return True, "llama-cpp-python hazır."

    def stop_worker(self) -> tuple[bool, str]:
        """Geriye uyumluluk. Sunucu veya modeli durdurur."""
        if self._use_llama_server and self._server_backend:
            return self._server_backend.stop_server()
        self._backend.unload()
        return True, "GGUF model bellekten çıkarıldı."

    def restart_worker(self) -> tuple[bool, str]:
        """Model boşalt, yeniden yüklenmeye hazır hale getir."""
        self._backend.unload()
        return True, "Worker sıfırlandı."

    # ─────────────────────── Model İşlemleri ────────────────────

    @property
    def is_loaded(self) -> bool:
        if self._use_llama_server and self._server_backend:
            return self._server_backend.is_loaded
        return self._backend.is_loaded

    @property
    def loaded_model_name(self) -> str:
        if self._use_llama_server and self._server_backend:
            return self._server_backend.loaded_model_name
        return self._backend.loaded_model_name

    def load(self, model_id: str, device: str = "cpu",
             session_id: str = "", n_ctx: int = 4096) -> tuple[bool, str]:
        """
        GGUF model yükle.

        model_id:
            • Tam dosya yolu: "C:/OpenVINO_LLM/gguf/qwen2.5-7b-q4.gguf"
            • HF formatı (yerel cache değil, uyarı verilir): "Qwen/Qwen2.5-7B-Instruct"

        device:
            "cpu"  → CPU çıkarımı
            "xpu"  → Intel Arc SYCL (SYCL derlemesi gerekir)
            "gpu"  → SYCL (xpu ile aynı)
            "auto" → llama.cpp cihaz kararına bırakır
        """
        self._device = device.lower()

        # llama-cpp-python kurulu değilse öneri ver
        if not self._backend.is_available():
            ok, msg = self.start_worker()
            if not ok:
                return False, msg

        # HF model_id mi yoksa dosya yolu mu?
        if not model_id.endswith(".gguf") and "/" in model_id:
            return False, (
                f"'{model_id}' bir HF model ID gibi görünüyor.\n"
                "LlamaCppBackend yalnızca yerel .gguf dosyaları çalıştırır.\n"
                "GGUF dosyasını indirmek için:\n"
                "  • HuggingFace'den manuel olarak .gguf dosyasını indirin\n"
                "  • Model Galerisi → GGUF İndir butonunu kullanın\n"
                "  • Veya Ollama backend ile direkt model_id kullanın"
            )

        # n_gpu_layers: cihaza göre otomatik ayarla
        n_gpu_layers = -1 if self._device in ("xpu", "gpu", "auto") else 0

        # n_ctx: kullanıcı vermediyse model adından tahmin et
        if n_ctx <= 512:
            n_ctx = self._backend._guess_context(Path(model_id).stem)
        n_ctx = max(n_ctx, 512)

        # llama-server backend: önce mevcut sunucuya bağlan, yoksa başlat
        if self._use_llama_server and self._server_backend:
            ok, msg = self._server_backend.start_server(
                exe_path=self.LLAMA_SERVER_EXE,
                model_path=model_id,
                device="SYCL0",
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                host=self.LLAMA_SERVER_HOST,
                port=self.LLAMA_SERVER_PORT,
            )
            if ok:
                return True, msg
            # Başarısız olduysa llama-cpp-python'a fallback
            logger.warning(f"llama-server başlatılamadı ({msg}), llama-cpp-python'a fallback.")

        return self._backend.load(
            model_path=model_id,
            device=self._device,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            session_id=session_id,
        )

    def generate(self, prompt: str, params: dict,
                 session_id: str = "", raw_prompt: str = "",
                 system_prompt: str = "",
                 history: list = None) -> tuple[str, dict]:
        """Metin üret."""
        if not self.is_loaded:
            return "", {"error": "Model yüklü değil."}
        if self._use_llama_server and self._server_backend and self._server_backend.is_loaded:
            return self._server_backend.generate(
                prompt=prompt, params=params,
                session_id=session_id, raw_prompt=raw_prompt,
                system_prompt=system_prompt, history=history,
            )
        return self._backend.generate(
            prompt=prompt, params=params,
            session_id=session_id, raw_prompt=raw_prompt,
            system_prompt=system_prompt, history=history,
        )

    def generate_stream(self, prompt: str, params: dict,
                        system_prompt: str = "",
                        history: list = None):
        """Streaming token üretimi."""
        if not self.is_loaded:
            yield ""
            return
        if self._use_llama_server and self._server_backend and self._server_backend.is_loaded:
            yield from self._server_backend.generate_stream(
                prompt=prompt, params=params,
                system_prompt=system_prompt, history=history
            )
        else:
            yield from self._backend.generate_stream(
                prompt=prompt, params=params,
                system_prompt=system_prompt, history=history
            )

    def generate_raw(self, prompt: str, params: dict) -> tuple[str, dict]:
        """Ham prompt ile üretim (DSPy sınıflandırması için)."""
        if not self.is_loaded:
            return "", {"error": "Model yüklü değil."}
        if self._use_llama_server and self._server_backend and self._server_backend.is_loaded:
            return self._server_backend.generate_raw(prompt=prompt, params=params)
        return self._backend.generate_raw(prompt=prompt, params=params)

    # ─────────────────────── Durum & Bellek ─────────────────────

    def get_status(self) -> dict:
        mem = _psutil_vm() if _psutil_ok() else None
        info = {
            "loaded":        self._backend.is_loaded,
            "model_id":      self._backend._loaded_id or "",
            "model_path":    self._backend._loaded_path or "",
            "device":        self._device,
            "xpu_available": self._backend.check_sycl_support(),
            "backend":       "llamacpp",
        }
        if mem is not None:
            m = mem
            info.update({
                "ram_total_gb": round(m.total    / 1024**3, 1),
                "ram_used_gb":  round(m.used     / 1024**3, 1),
                "ram_avail_gb": round(m.available / 1024**3, 1),
            })
        return {"ok": True, "result": info}

    def get_memory_info(self) -> dict:
        return self._backend.get_memory_info()

    def scan_local_gguf(self) -> list:
        """Yerel GGUF dosyalarını tara (orchestrator için)."""
        return self._backend.scan_gguf_models()


def _psutil_ok() -> bool:
    try:
        import psutil  # noqa: F401
        return True
    except ImportError:
        return False


def _psutil_vm():
    import psutil
    return psutil.virtual_memory()