"""
modules/ipex_backend.py
Ollama ve llama.cpp/SYCL backend desteği.

IPEX-LLM Durumu (Mart 2026):
  - ipex-llm projesi EOL (End-of-Life) ilan edildi.
  - ipex-llm'in XPU/SYCL optimizasyonları doğrudan PyTorch ana dalına
    (2.4+) aktarıldı (upstream).
  - Önerilen geçiş yolu:
      a) Ollama Portable (IPEX-LLM fork) → Intel iGPU desteği için
         https://github.com/ipex-llm/ollama (IPEX-LLM Ollama fork)
      b) llama-cpp-python SYCL backend → doğrudan GGUF model çalıştırma
         pip install llama-cpp-python --extra-index-url \
           https://abetlen.github.io/llama-cpp-python/whl/cu124  (SYCL için derleme gerekir)

Bu modül iki backend sağlar:
  1. OllamaBackend  — standart Ollama API (localhost:11434)
                     IPEX-LLM Ollama fork ile iGPU hızlandırma mümkün
  2. LlamaCppBackend — llama-cpp-python ile yerel GGUF çalıştırma
                       CPU/SYCL(XPU) cihaz desteği

Kurulum:
  Ollama (standart):     https://ollama.com/download
  Ollama (IPEX fork):    https://github.com/ipex-llm/ollama/releases
                         → start_ollama.bat içinde OLLAMA_NUM_GPU=999 ayarla
  llama-cpp-python:      pip install llama-cpp-python
                         (SYCL/XPU için kaynak derlemesi gerekir)
"""

import os
import time
import json
import logging
import threading
import gc
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

import psutil
import requests

logger = logging.getLogger(__name__)

from core.config import GGUF_DIR, OPENVINO_LLM_HOME, LLAMA_SERVER_EXE

# ─────────────────────────────────────────
# Veri sınıfları
# ─────────────────────────────────────────

@dataclass
class IPEXModelInfo:
    name: str                        # Görünen ad
    source: str                      # "ollama" | "llamacpp" | "llamacpp_hf"
    model_id: str                    # Ollama model adı veya GGUF dosya yolu
    size_gb: float = 0.0
    context_length: int = 4096
    tags: list = field(default_factory=list)


# ─────────────────────────────────────────
# Ollama Backend
# ─────────────────────────────────────────

class OllamaBackend:
    """
    Ollama REST API üzerinden yerel LLM çağrısı.

    Standart Ollama veya IPEX-LLM Ollama fork ile çalışır.
    IPEX fork kullanılıyorsa Intel Arc iGPU otomatik devreye girer;
    bu durumda backend_type "ipex_ollama" olarak raporlanır.
    """

    def __init__(self, base_url: str = "http://localhost:11434", db_manager=None):
        self.base_url = base_url.rstrip("/")
        self.db = db_manager
        self._current_model: Optional[str] = None
        self._lock = threading.Lock()

    # ── Durum ──────────────────────────────────────────

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def detect_backend_type(self) -> str:
        """
        Çalışan Ollama'nın standart mı yoksa IPEX-LLM fork mu
        olduğunu anlamaya çalışır. IPEX fork genellikle /api/version
        yanıtında 'ipex' veya 'intel' içerir; yoksa 'ollama' döner.
        """
        try:
            r = requests.get(f"{self.base_url}/api/version", timeout=3)
            if r.status_code == 200:
                ver = r.json().get("version", "").lower()
                if "ipex" in ver or "intel" in ver:
                    return "ipex_ollama"
        except Exception:
            pass
        return "ollama"

    # ── Model İşlemleri ────────────────────────────────

    def list_models(self) -> list[IPEXModelInfo]:
        """Ollama'da yüklü modelleri döndür."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
            models = []
            for m in r.json().get("models", []):
                size_gb = m.get("size", 0) / (1024 ** 3)
                models.append(IPEXModelInfo(
                    name=f"[Ollama] {m['name']}",
                    source="ollama",
                    model_id=m["name"],
                    size_gb=round(size_gb, 1),
                    context_length=m.get("details", {}).get("context_length", 4096),
                    tags=m.get("details", {}).get("families", []),
                ))
            return models
        except Exception as e:
            logger.warning(f"Ollama model listesi alınamadı: {e}")
            return []

    def pull_model(self, model_id: str) -> tuple[bool, str]:
        """Ollama'ya model indir (streaming progress)."""
        try:
            r = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_id},
                stream=True, timeout=600
            )
            for line in r.iter_lines():
                if line:
                    data = json.loads(line)
                    if data.get("status") == "success":
                        return True, f"'{model_id}' başarıyla indirildi."
            return True, f"'{model_id}' hazır."
        except Exception as e:
            return False, f"Pull hatası: {e}"

    def load(self, model_id: str, session_id: str = "") -> tuple[bool, str]:
        """Model seç (Ollama'da önceden yüklü olması gerekir)."""
        if not self.is_available():
            return False, (
                "Ollama çalışmıyor. Şunlardan birini başlatın:\n"
                "  • Standart: ollama serve\n"
                "  • IPEX fork: C:\\ipex-ollama\\start_ollama.bat"
            )
        models = [m.model_id for m in self.list_models()]
        if model_id not in models:
            return False, (
                f"'{model_id}' Ollama'da bulunamadı.\n"
                f"İndirmek için: ollama pull {model_id}"
            )
        self._current_model = model_id
        if self.db:
            btype = self.detect_backend_type()
            self.db.log_general(session_id, "INFO", "OllamaBackend",
                                f"Model seçildi: {model_id} [{btype}]",
                                {"backend": btype})
        return True, f"Ollama modeli hazır: {model_id}"

    # ── Üretim ─────────────────────────────────────────

    def generate(self, prompt: str, params: dict,
                 session_id: str = "", raw_prompt: str = "",
                 system_prompt: str = "") -> tuple[str, dict]:
        if not self._current_model:
            return "", {"error": "Ollama modeli seçilmedi."}

        with self._lock:
            start = time.time()
            try:
                payload = {
                    "model": self._current_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": params.get("temperature", 0.7),
                        "num_predict": params.get("max_tokens", 512),
                        "top_p": params.get("top_p", 0.9),
                        "top_k": params.get("top_k", 50),
                        "repeat_penalty": params.get("repetition_penalty", 1.1),
                    }
                }
                if system_prompt:
                    payload["system"] = system_prompt

                r = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=300
                )
                r.raise_for_status()
                data = r.json()
                response = data.get("response", "").strip()

                elapsed_ms = (time.time() - start) * 1000
                eval_count = data.get("eval_count", len(response.split()))
                prompt_eval_count = data.get("prompt_eval_count", len(prompt.split()))
                tps = eval_count / max(elapsed_ms / 1000, 0.001)

                metrics = {
                    "duration_ms": round(elapsed_ms, 1),
                    "input_tokens": prompt_eval_count,
                    "output_tokens": eval_count,
                    "tokens_per_second": round(tps, 2),
                }

                if self.db:
                    self.db.log_llm(
                        session_id=session_id,
                        model_name=f"ollama/{self._current_model}",
                        model_type="ollama",
                        params=params,
                        system_prompt=system_prompt,
                        final_prompt=prompt,
                        raw_prompt=raw_prompt,
                        response=response,
                        input_tokens=prompt_eval_count,
                        output_tokens=eval_count,
                        duration_ms=elapsed_ms,
                        tokens_per_second=tps,
                    )

                return response, metrics

            except requests.Timeout:
                err = "Ollama timeout — model çok uzun sürdü."
                logger.error(err)
                return "", {"error": err}
            except Exception as e:
                logger.error(f"Ollama generate hatası: {e}")
                if self.db:
                    self.db.log_error(session_id, "OllamaBackend.generate", e)
                return "", {"error": str(e)}

    def generate_raw(self, prompt: str, params: dict) -> tuple[str, dict]:
        """Ham prompt ile üretim (system mesajı yok). DSPy sınıflandırması için."""
        if not self._current_model:
            return "", {"error": "Ollama modeli seçilmedi."}

        with self._lock:
            start = time.time()
            try:
                payload = {
                    "model": self._current_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": params.get("temperature", 0.0),
                        "num_predict": params.get("max_tokens", 50),
                        "top_p": 1.0,
                        "top_k": 50,
                        "repeat_penalty": 1.0,
                    }
                }
                r = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=30
                )
                r.raise_for_status()
                data = r.json()
                response = data.get("response", "").strip()
                elapsed_ms = (time.time() - start) * 1000
                return response, {"duration_ms": elapsed_ms}
            except Exception as e:
                logger.error(f"Ollama generate_raw hatası: {e}")
                return "", {"error": str(e)}

    # ── Özellikler ─────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._current_model is not None and self.is_available()

    @property
    def loaded_model_name(self) -> str:
        return f"ollama/{self._current_model}" if self._current_model else ""

    def get_memory_info(self) -> dict:
        mem = psutil.virtual_memory()
        return {
            "total_gb":    round(mem.total    / 1024**3, 1),
            "used_gb":     round(mem.used     / 1024**3, 1),
            "available_gb": round(mem.available / 1024**3, 1),
            "percent":     mem.percent,
        }


# ─────────────────────────────────────────
# llama-cpp-python Backend (GGUF / SYCL)
# ─────────────────────────────────────────

class LlamaCppBackend:
    """
    llama-cpp-python ile GGUF modeli doğrudan çalıştırır.

    Desteklenen cihazlar:
      - CPU (varsayılan)
      - Intel Arc XPU / iGPU: SYCL derlemesi gerektirir.
        https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md

    Kurulum:
      CPU (hazır wheel):
        pip install llama-cpp-python

      SYCL/XPU (Intel Arc):
        # Intel oneAPI Base Toolkit kurulu olmalı
        CMAKE_ARGS="-DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx" \
          pip install llama-cpp-python --no-binary llama-cpp-python

      Windows (SYCL):
        set CMAKE_ARGS=-DGGML_SYCL=ON
        pip install llama-cpp-python --no-binary llama-cpp-python

    GGUF model dizini: C:\\OpenVINO_LLM\\gguf\\ (otomatik taranır)
    """

    # GGUF model arama yolları
    # Not: OPENVINO_LLM_HOME (C:\OpenVINO_LLM) kasıtlı olarak çıkarıldı.
    # O dizinin tamamını rglob ile taramak yavaş ve gereksiz; .gguf dosyaları
    # GGUF_DIR (C:\OpenVINO_LLM\gguf) altına konulmalıdır.
    SEARCH_PATHS = [
        GGUF_DIR,
        Path.home() / ".cache" / "huggingface" / "hub",
    ]

    def __init__(self, db_manager=None):
        self.db = db_manager
        self._model = None           # llama_cpp.Llama örneği
        self._loaded_path: Optional[str] = None
        self._loaded_id: Optional[str] = None
        self._device: str = "cpu"
        self._lock = threading.Lock()
        self._available = self._check_llama_cpp()

    # ── Kurulum Kontrolü ───────────────────────────────

    def _check_llama_cpp(self) -> bool:
        try:
            import llama_cpp  # noqa
            logger.info("llama-cpp-python mevcut.")
            return True
        except ImportError:
            logger.info("llama-cpp-python kurulu değil. pip install llama-cpp-python")
            return False

    def is_available(self) -> bool:
        return self._available

    def check_sycl_support(self) -> bool:
        """
        llama-cpp-python'un SYCL/XPU desteği ile derlenip derlenmediğini kontrol et.

        Yöntem sırası:
          1. llama_cpp._lib sembol kontrolü (ggml_sycl_can_generate, ggml_backend_sycl_init)
          2. llama_print_system_info() çıktısında "SYCL" arama
          3. Llama(n_gpu_layers=1) ile başlangıç testi — SYCL varsa GPU cihazı listelenir
        """
        try:
            import llama_cpp

            # Yöntem 1: Kütüphane sembolleri
            lib = getattr(llama_cpp, "_lib", None)
            if lib is not None:
                for sym in ("ggml_sycl_can_generate", "ggml_backend_sycl_init",
                            "ggml_sycl_init", "ggml_backend_sycl_buffer_type"):
                    if hasattr(lib, sym):
                        logger.debug(f"SYCL sembol bulundu: {sym}")
                        return True

            # Yöntem 2: system_info string
            if hasattr(llama_cpp, "llama_print_system_info"):
                try:
                    info = llama_cpp.llama_print_system_info()
                    if isinstance(info, bytes):
                        info = info.decode("utf-8", errors="ignore")
                    if "SYCL" in info or "XPU" in info:
                        logger.debug(f"SYCL system_info'da bulundu: {info[:200]}")
                        return True
                except Exception:
                    pass

            # Yöntem 3: llama_cpp stdout'unu yakala — "SYCL" veya "Arc" geçiyor mu?
            # Model yüklendiğinde "using device SYCL0 (Intel Arc)" gibi çıktı üretir
            # Bu çıktı daha önce logda göründü → SYCL derlemesi başarılı
            # Kontrol: llama_cpp.__version__ veya build metadata'da SYCL var mı?
            try:
                import subprocess, sys
                result = subprocess.run(
                    [sys.executable, "-c",
                     "import llama_cpp; m=llama_cpp.Llama.__module__; "
                     "import importlib; mod=importlib.import_module(m); "
                     "print(getattr(mod, '__file__', ''))"],
                    capture_output=True, text=True, timeout=5
                )
                # llama_cpp DLL path'i al, içinde sycl8.dll var mı kontrol et
                dll_dir = ""
                lib_obj = getattr(llama_cpp, "_lib", None)
                if lib_obj and hasattr(lib_obj, "_name"):
                    dll_dir = str(lib_obj._name)
                if dll_dir:
                    import os
                    parent = os.path.dirname(dll_dir)
                    sycl_dlls = [f for f in os.listdir(parent)
                                 if "sycl" in f.lower() or "ggml-sycl" in f.lower()]
                    if sycl_dlls:
                        logger.debug(f"SYCL DLL bulundu: {sycl_dlls}")
                        return True
            except Exception:
                pass

            return False

        except Exception:
            return False

    # ── GGUF Model Tarama ──────────────────────────────

    def scan_gguf_models(self) -> list[IPEXModelInfo]:
        """
        Tanımlı dizinlerde .gguf dosyalarını tara.
        Bulunan her dosya bir IPEXModelInfo olarak döner.
        """
        found = []
        seen = set()

        for search_dir in self.SEARCH_PATHS:
            if not search_dir.exists():
                continue
            try:
                gguf_files = list(search_dir.rglob("*.gguf"))
            except OSError as e:
                logger.warning(f"GGUF tarama hatası ({search_dir}): {e}")
                continue
            for gguf_file in gguf_files:
                path_str = str(gguf_file)
                if path_str in seen:
                    continue
                seen.add(path_str)

                try:
                    size_gb = round(gguf_file.stat().st_size / 1024**3, 1)
                except OSError:
                    size_gb = 0.0

                name = gguf_file.stem  # dosya adı, uzantısız
                ctx  = self._guess_context(name)

                found.append(IPEXModelInfo(
                    name=f"[GGUF] {name}",
                    source="llamacpp",
                    model_id=path_str,
                    size_gb=size_gb,
                    context_length=ctx,
                    tags=["gguf", "yerel"],
                ))

        if not found:
            logger.info("GGUF model bulunamadı.")
        else:
            logger.info(f"{len(found)} GGUF model bulundu.")
        return found

    def _guess_context(self, name: str) -> int:
        """Model adından bağlam uzunluğunu tahmin et."""
        nl = name.lower()
        for pattern, ctx in [
            ("128k", 131072), ("32k", 32768), ("16k", 16384),
            ("8k", 8192), ("4k", 4096),
        ]:
            if pattern in nl:
                return ctx
        # Bilinen mimariler için varsayılan
        if any(k in nl for k in ["llama3.1", "llama-3.1", "qwen2.5", "mistral-nemo"]):
            return 32768
        if any(k in nl for k in ["phi3.5", "phi-3.5", "llama3.2"]):
            return 131072
        return 4096

    # ── Model Yükleme ──────────────────────────────────

    def load(self, model_path: str, device: str = "cpu",
             n_ctx: int = 4096, n_gpu_layers: int = 0,
             session_id: str = "") -> tuple[bool, str]:
        """
        GGUF modelini yükle.

        Parametreler:
            model_path  : .gguf dosyasının tam yolu veya model_id
            device      : "cpu" | "xpu" | "gpu"
            n_ctx       : bağlam penceresi (token sayısı)
            n_gpu_layers: GPU'ya taşınacak katman sayısı
                         -1 = tümü, 0 = CPU only
                         SYCL/XPU için: -1 veya yüksek bir sayı
        """
        if not self._available:
            return False, (
                "llama-cpp-python kurulu değil.\n"
                "CPU için: pip install llama-cpp-python\n"
                "Intel Arc XPU için SYCL derlemesi gerekir.\n"
                "Alternatif: Ollama backend kullanın (IPEX Ollama fork ile iGPU desteği)."
            )

        # Bellek kontrolü
        mem = psutil.virtual_memory()
        if mem.available / 1024**3 < 2.0:
            return False, "Yetersiz RAM (en az 2 GB gerekli)."

        with self._lock:
            try:
                self._unload_internal()

                import llama_cpp
                import psutil as _ps

                # Cihaz ayarları
                effective_device = device.lower()
                _n_gpu_layers    = n_gpu_layers

                # SYCL/XPU kontrolü: derleme yoksa CPU'ya düş
                sycl_ok = self.check_sycl_support()
                if effective_device in ("xpu", "gpu", "auto"):
                    if sycl_ok:
                        if _n_gpu_layers == 0:
                            _n_gpu_layers = -1
                        logger.info(f"SYCL/XPU mevcut. GPU modunda yükleniyor "
                                    f"(n_gpu_layers={_n_gpu_layers}).")
                    else:
                        logger.warning(
                            "SYCL/XPU desteği YOK — CPU'ya fallback yapılıyor. "
                            "Intel Arc GPU hızlandırması için llama-cpp-python'u "
                            "SYCL ile yeniden derlemeniz gerekiyor."
                        )
                        effective_device = "cpu"
                        _n_gpu_layers    = 0
                else:
                    _n_gpu_layers = 0  # CPU only

                self._device = effective_device

                # RAM ölçümü: yükleme öncesi
                mem_before = _ps.virtual_memory().used

                logger.info(f"GGUF model yükleniyor: {model_path} "
                            f"[device={effective_device}, n_ctx={n_ctx}, "
                            f"n_gpu_layers={_n_gpu_layers}]")

                # verbose=True → llama.cpp'in kendi yükleme çıktısını logla
                # (sessiz hatalarda nedenini anlamak için önemli)
                self._model = llama_cpp.Llama(
                    model_path=model_path,
                    n_ctx=n_ctx,
                    n_gpu_layers=_n_gpu_layers,
                    verbose=True,           # yükleme loglarını göster
                    n_threads=None,         # otomatik thread sayısı
                )

                # RAM ölçümü: yükleme sonrası — gerçekten yüklendi mi?
                mem_after  = _ps.virtual_memory().used
                ram_delta  = (mem_after - mem_before) / 1024**3  # GB
                model_size = Path(model_path).stat().st_size / 1024**3

                if ram_delta < 0.5:
                    # Beklenmedik: model boyutuna göre en az 500 MB artış olmalı
                    logger.warning(
                        f"RAM artışı çok düşük ({ram_delta:.2f} GB) — "
                        f"model {model_size:.1f} GB. "
                        "Model gerçekten yüklenmemiş olabilir!"
                    )

                self._loaded_path = model_path
                self._loaded_id   = Path(model_path).stem

                if self.db and session_id:
                    self.db.log_general(
                        session_id, "INFO", "LlamaCppBackend",
                        f"GGUF model yüklendi: {self._loaded_id}",
                        {"device": effective_device, "n_ctx": n_ctx,
                         "n_gpu_layers": _n_gpu_layers,
                         "ram_delta_gb": round(ram_delta, 2),
                         "sycl_available": sycl_ok}
                    )

                warn = ""
                if effective_device == "cpu" and device.lower() in ("xpu", "gpu", "auto"):
                    warn = " ⚠️ SYCL yok → CPU modunda çalışıyor"

                device_str = effective_device.upper()
                return True, (
                    f"✅ GGUF yüklendi: {self._loaded_id} "
                    f"[{device_str}, ctx={n_ctx}, RAM +{ram_delta:.1f} GB]"
                    f"{warn}"
                )

            except Exception as e:
                logger.error(f"GGUF yükleme hatası: {e}", exc_info=True)
                self._unload_internal()
                if self.db and session_id:
                    self.db.log_error(session_id, "LlamaCppBackend.load", e)
                return False, f"❌ GGUF yükleme hatası: {type(e).__name__}: {e}"

    # ── Model Boşaltma ─────────────────────────────────

    def unload(self) -> tuple[bool, str]:
        with self._lock:
            self._unload_internal()
        return True, "GGUF model bellekten çıkarıldı."

    def _unload_internal(self):
        if self._model is not None:
            try:
                del self._model
            except Exception:
                pass
        self._model      = None
        self._loaded_path = None
        self._loaded_id  = None
        gc.collect()
        logger.info("GGUF model bellekten temizlendi.")

    # ── Üretim ─────────────────────────────────────────

    def _apply_gguf_chat_template(self, system_prompt: str, user_prompt: str) -> str:
        """
        GGUF metadata'daki Jinja2 chat template'ini uygula.
        llama_cpp bunu create_chat_completion'da kısmen destekler ama
        özel token formatlarında (gpt-oss gibi) başarısız olabilir.
        Bu metot template'i direkt llama_cpp.llama_chat_format modülü ile render eder.
        """
        try:
            # llama_cpp kendi içinde jinja2 chat template render edebilir:
            # _model.metadata'dan template al, llama_cpp.llama_chat_apply_template ile uygula
            metadata = getattr(self._model, "metadata", {}) or {}
            tmpl = metadata.get("tokenizer.chat_template", "")

            if not tmpl:
                # Fallback: template yoksa basit format
                return self._simple_chat_format(system_prompt, user_prompt)

            # llama_cpp.llama_chat_apply_template (0.3.x+) dene
            import llama_cpp
            if hasattr(llama_cpp, "llama_chat_apply_template"):
                # messages list hazırla
                msgs = []
                if system_prompt:
                    msgs.append({"role": "system", "content": system_prompt})
                msgs.append({"role": "user", "content": user_prompt})
                # add_generation_prompt=True → modelin yanıtlamaya başlaması için
                try:
                    result = llama_cpp.llama_chat_apply_template(
                        self._model._model,
                        tmpl.encode("utf-8"),
                        msgs,
                        True,  # add_ass (add generation prompt)
                    )
                    if result and len(result) > 10:
                        logger.debug(f"Jinja2 template uygulandı, uzunluk: {len(result)}")
                        return result
                except Exception as e:
                    logger.warning(f"llama_chat_apply_template başarısız: {e}")

            # Fallback: create_chat_completion'ın tokenizer'ını kullan (metadata template dahil)
            return self._simple_chat_format(system_prompt, user_prompt)

        except Exception as e:
            logger.warning(f"Chat template uygulama hatası: {e}")
            return self._simple_chat_format(system_prompt, user_prompt)

    def _simple_chat_format(self, system_prompt: str, user_prompt: str) -> str:
        """Basit fallback chat format — template yoksa kullanılır."""
        sys_msg = system_prompt or "You are a helpful assistant."
        return f"{sys_msg}\n\nUser: {user_prompt}\nAssistant:"

    def generate(self, prompt: str, params: dict,
                 session_id: str = "", raw_prompt: str = "",
                 system_prompt: str = "") -> tuple[str, dict]:
        """
        Metin üret.

        Strateji (öncelik sırasıyla):
        1. create_chat_completion — llama_cpp'in kendi chat template uygulaması.
           GGUF metadata'daki template'i kullanır (gpt-oss dahil).
        2. Hata alınırsa create_completion ile ham prompt formatı.
        """
        if self._model is None:
            return "", {"error": "GGUF model yüklü değil."}

        with self._lock:
            start = time.time()
            try:
                sys_msg = system_prompt or "You are a helpful assistant."
                messages = [
                    {"role": "system", "content": sys_msg},
                    {"role": "user",   "content": prompt},
                ]

                temperature = params.get("temperature", 0.7)
                max_tokens  = params.get("max_tokens", 512)
                top_p       = params.get("top_p", 0.9)
                top_k       = params.get("top_k", 50)
                repeat_pen  = params.get("repetition_penalty", 1.1)

                response_text = ""

                # ── Yöntem 1: create_chat_completion ──────────────────────
                # llama_cpp 0.3.x, GGUF metadata'daki Jinja2 template'i
                # otomatik kullanır. gpt-oss için de doğru formatı üretir.
                try:
                    result = self._model.create_chat_completion(
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repeat_penalty=repeat_pen,
                    )
                    content = result["choices"][0]["message"].get("content") or ""
                    # gpt-oss bazen content=None, reasoning content ayrı gelir
                    if not content:
                        # tool_calls veya thinking field dene
                        content = (result["choices"][0].get("message", {})
                                   .get("reasoning_content", ""))
                    response_text = content.strip()

                    elapsed_ms    = (time.time() - start) * 1000
                    usage         = result.get("usage", {})
                    input_tokens  = usage.get("prompt_tokens",  len(prompt.split()))
                    output_tokens = usage.get("completion_tokens", len(response_text.split()))

                except Exception as e1:
                    logger.warning(f"create_chat_completion başarısız ({e1}), "
                                   "create_completion deneniyor...")
                    # ── Yöntem 2: create_completion (ham prompt) ───────────
                    # GGUF'un kendi tokenizer'ına bırakıyoruz
                    try:
                        formatted = self._apply_gguf_chat_template(sys_msg, prompt)
                        result2 = self._model.create_completion(
                            prompt=formatted,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            repeat_penalty=repeat_pen,
                            stop=["<|end|>", "<|return|>", "<|endoftext|>",
                                  "\nUser:", "\nHuman:"],
                        )
                        response_text = result2["choices"][0]["text"].strip()
                        elapsed_ms    = (time.time() - start) * 1000
                        input_tokens  = result2.get("usage", {}).get(
                            "prompt_tokens", len(formatted.split()))
                        output_tokens = result2.get("usage", {}).get(
                            "completion_tokens", len(response_text.split()))
                    except Exception as e2:
                        logger.error(f"create_completion da başarısız: {e2}", exc_info=True)
                        raise e2

                tps = output_tokens / max(elapsed_ms / 1000, 0.001)
                metrics = {
                    "duration_ms":       round(elapsed_ms, 1),
                    "input_tokens":      input_tokens,
                    "output_tokens":     output_tokens,
                    "tokens_per_second": round(tps, 2),
                }

                if self.db:
                    self.db.log_llm(
                        session_id=session_id,
                        model_name=f"llamacpp/{self._loaded_id}",
                        model_type="llamacpp",
                        params=params,
                        system_prompt=system_prompt,
                        final_prompt=prompt,
                        raw_prompt=raw_prompt,
                        response=response_text,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        duration_ms=elapsed_ms,
                        tokens_per_second=tps,
                    )

                return response_text, metrics

            except Exception as e:
                logger.error(f"GGUF generate hatası: {e}", exc_info=True)
                if self.db:
                    self.db.log_error(session_id, "LlamaCppBackend.generate", e)
                return "", {"error": str(e)}

    def generate_raw(self, prompt: str, params: dict) -> tuple[str, dict]:
        """Ham prompt ile üretim (DSPy sınıflandırması için)."""
        if self._model is None:
            return "", {"error": "GGUF model yüklü değil."}

        with self._lock:
            start = time.time()
            try:
                # Model mimarisine göre stop tokenları
                stop_tokens = ["<|end|>", "<|return|>", "<|endoftext|>",
                               "\nUser:", "\nHuman:", "\n\n"]
                result = self._model(
                    prompt,
                    max_tokens=params.get("max_tokens", 50),
                    temperature=params.get("temperature", 0.0),
                    top_p=1.0,
                    top_k=1,
                    repeat_penalty=1.0,
                    echo=False,
                    stop=stop_tokens,
                )
                response  = result["choices"][0]["text"].strip()
                elapsed_ms = (time.time() - start) * 1000
                return response, {"duration_ms": elapsed_ms}
            except Exception as e:
                logger.error(f"GGUF generate_raw hatası: {e}")
                return "", {"error": str(e)}

    # ── Özellikler ─────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def loaded_model_name(self) -> str:
        if not self._loaded_id:
            return ""
        device_tag = self._device.upper() if self._device else "CPU"
        sycl_ok = self.check_sycl_support()
        if device_tag == "CPU" and sycl_ok:
            device_tag = "SYCL/XPU"
        return f"llamacpp/{self._loaded_id} [{device_tag}]"

    def get_memory_info(self) -> dict:
        mem = psutil.virtual_memory()
        return {
            "total_gb":    round(mem.total    / 1024**3, 1),
            "used_gb":     round(mem.used     / 1024**3, 1),
            "available_gb": round(mem.available / 1024**3, 1),
            "percent":     mem.percent,
        }


# ─────────────────────────────────────────
# Geriye Dönük Uyumluluk Uyarısı
# ─────────────────────────────────────────

class IPEXHFBackend:
    """
    KALDIRILDI — ipex-llm Mart 2026'da EOL ilan edildi.

    Kullanılabilir alternatifler:
      1. OllamaBackend  + IPEX-LLM Ollama fork (iGPU hızlandırma)
      2. LlamaCppBackend + SYCL derlemesi (doğrudan GGUF çalıştırma)

    Bu sınıf yalnızca eski kodu kırmamak için tutulmuştur;
    hiçbir işlem yapmaz ve kullanımda hata döndürür.
    """

    def __init__(self, *args, **kwargs):
        logger.warning(
            "IPEXHFBackend kullanımdan kaldırıldı (ipex-llm EOL, Mart 2026). "
            "LlamaCppBackend veya OllamaBackend kullanın."
        )

    def load(self, *args, **kwargs) -> tuple[bool, str]:
        return False, (
            "IPEXHFBackend kaldırıldı.\n"
            "ipex-llm projesi Mart 2026'da EOL ilan edildi.\n"
            "Alternatifler:\n"
            "  • Ollama + IPEX Ollama fork (iGPU hızlandırma)\n"
            "  • llama-cpp-python SYCL backend\n"
            "Lütfen Ollama veya LlamaCpp backend'ini kullanın."
        )

    def generate(self, *args, **kwargs) -> tuple[str, dict]:
        return "", {"error": "IPEXHFBackend kaldırıldı. Ollama veya LlamaCpp kullanın."}

    def generate_raw(self, *args, **kwargs) -> tuple[str, dict]:
        return "", {"error": "IPEXHFBackend kaldırıldı."}

    @property
    def is_loaded(self) -> bool:
        return False

    @property
    def loaded_model_name(self) -> str:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# LlamaServerBackend — llama-server.exe HTTP API istemcisi
# ─────────────────────────────────────────────────────────────────────────────

class LlamaServerBackend:
    """
    Harici llama-server.exe sürecine HTTP üzerinden bağlanan backend.

    llama-server, llama.cpp'nin resmi C++ sunucusudur:
      - OpenAI uyumlu API (/v1/chat/completions, /v1/completions)
      - SYCL/Intel Arc GPU desteği (llama-cpp-python gerektirmez)
      - Qwen3.5 dahil en yeni mimariler desteklenir

    Kullanım:
        backend = LlamaServerBackend()
        ok, msg = backend.start_server(
            exe_path="C:/OpenVINO_LLM/llama-server/llama-server.exe",
            model_path="C:/OpenVINO_LLM/gguf/Qwen3.5-9B-Q4_K_M.gguf",
            device="SYCL0", n_gpu_layers=99, n_ctx=4096
        )
        text, metrics = backend.generate(prompt, params)
        backend.stop_server()
    """

    DEFAULT_HOST = "127.0.0.1"
    DEFAULT_PORT = 8080
    DEFAULT_EXE  = str(LLAMA_SERVER_EXE)

    def __init__(self, db_manager=None):
        self.db          = db_manager
        self._host       = self.DEFAULT_HOST
        self._port       = self.DEFAULT_PORT
        self._base_url   = f"http://{self._host}:{self._port}"
        self._process    = None          # subprocess.Popen
        self._model_path = ""
        self._model_name = ""
        self._device     = "SYCL0"
        self._is_loaded  = False
        self._exe_path   = self.DEFAULT_EXE
        self._lock       = threading.Lock()

    # ── Sunucu Yönetimi ──────────────────────────────────────────

    def start_server(self,
                     exe_path:     str = "",
                     model_path:   str = "",
                     device:       str = "SYCL0",
                     n_gpu_layers: int = 99,
                     n_ctx:        int = 4096,
                     host:         str = DEFAULT_HOST,
                     port:         int = DEFAULT_PORT,
                     extra_args:   list = None) -> tuple[bool, str]:
        """
        llama-server.exe'yi verilen modelle başlat veya yeniden başlat.

        - Eğer aynı model zaten çalışıyorsa bağlan, dokunma.
        - Farklı model istendiyse mevcut süreci öldür, yeni modelle başlat.
        - subprocess.DETACHED_PROCESS ile pipe tıkanması sorunu yok.
        """
        import subprocess

        self._host     = host
        self._port     = port
        self._base_url = f"http://{host}:{port}"
        self._exe_path = exe_path or self.DEFAULT_EXE
        self._device   = device

        # Sunucu çalışıyor mu?
        if self._ping_server():
            # Aynı model mi?
            if self._model_path == model_path:
                logger.info(f"LlamaServerBackend: aynı model zaten yüklü, bağlanıldı.")
                self._is_loaded = True
                return True, f"✅ llama-server bağlandı (zaten çalışıyor): {self._model_name}"
            # Farklı model — kapat, yeniden başlat
            logger.info(f"LlamaServerBackend: model değişti, sunucu yeniden başlatılıyor.")
            self._kill_server_on_port()

        if not Path(self._exe_path).exists():
            return False, (
                f"llama-server.exe bulunamadi: {self._exe_path} — C:\\OpenVINO_LLM\\llama-server\\ dizinini kontrol edin."
            )

        # CREATE_NEW_CONSOLE: ayrı pencerede aç, pipe yok, SYCL donma sorunu yok
        # NOT: DETACHED_PROCESS + CREATE_NEW_CONSOLE birlikte kullanılamaz (WinError 87)
        CREATE_NEW_CONSOLE = 0x00000010

        cmd = [
            self._exe_path,
            "-m", model_path,
            "--host", host,
            "--port", str(port),
            "-ngl", str(n_gpu_layers),
            "-c", str(n_ctx),
            "--device", device,
        ]
        if extra_args:
            cmd.extend(extra_args)

        logger.info(f"LlamaServerBackend: başlatılıyor → {Path(model_path).name}")
        try:
            self._process = subprocess.Popen(
                cmd,
                cwd=str(Path(self._exe_path).parent),
                creationflags=CREATE_NEW_CONSOLE,
            )
        except Exception as e:
            return False, f"llama-server başlatma hatası: {e}"

        # Hazır olana kadar arka planda bekle (UI donmasın)
        import time
        t_start  = time.time()
        deadline = t_start + 180
        while time.time() < deadline:
            if self._process.poll() is not None:
                return False, f"llama-server erken çıktı (kod={self._process.returncode})"
            if self._ping_server():
                self._is_loaded  = True
                self._model_path = model_path
                self._model_name = Path(model_path).stem
                elapsed = round(time.time() - t_start, 1)
                logger.info(f"LlamaServerBackend: hazır ({elapsed}s) — {self._model_name}")
                return True, f"✅ llama-server hazır ({elapsed}s) | {self._model_name} | {device}"
            time.sleep(2)

        self._process.kill()
        return False, "llama-server 180s içinde hazır olmadı. GPU sürücüsünü kontrol edin."

    def _kill_server_on_port(self):
        """Portu dinleyen süreci öldür (model değişimi için)."""
        import subprocess
        try:
            # netstat ile PID bul
            result = subprocess.run(
                ["netstat", "-ano"],
                capture_output=True, text=True
            )
            for line in result.stdout.splitlines():
                if f":{self._port}" in line and "LISTENING" in line:
                    pid = int(line.strip().split()[-1])
                    subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                                   capture_output=True)
                    logger.info(f"LlamaServerBackend: PID {pid} sonlandırıldı.")
                    import time; time.sleep(2)
                    break
        except Exception as e:
            logger.warning(f"_kill_server_on_port hatası: {e}")
        self._is_loaded = False
        self._process   = None

    def stop_server(self) -> tuple[bool, str]:
        """Sunucuyu durdur."""
        self._is_loaded = False
        # Önce kendi başlattığımız süreci dene
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except Exception:
                self._process.kill()
            self._process = None
            logger.info("LlamaServerBackend: sunucu durduruldu.")
            return True, "llama-server durduruldu."
        # Dışarıdan başlatılmış olabilir — port üzerinden öldür
        self._kill_server_on_port()
        return True, "llama-server durduruldu."

    def _ping_server(self) -> bool:
        """Sunucu /health endpoint'ine bağlanabilir mi?"""
        try:
            r = requests.get(f"{self._base_url}/health", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    # ── Durum ────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded and self._ping_server()

    @property
    def loaded_model_name(self) -> str:
        if not self._is_loaded:
            return ""
        return f"llama-server/{self._model_name} [SYCL0/Arc GPU]"

    def is_available(self) -> bool:
        return Path(self._exe_path).exists() or self._ping_server()

    # ── Üretim ───────────────────────────────────────────────────

    def generate(self, prompt: str, params: dict,
                 session_id: str = "", raw_prompt: str = "",
                 system_prompt: str = "") -> tuple[str, dict]:
        """
        /v1/chat/completions endpoint'ini kullan (OpenAI uyumlu).
        """
        t0 = time.time()

        max_tokens  = int(params.get("max_tokens",  512))
        temperature = float(params.get("temperature", 0.7))
        top_p       = float(params.get("top_p",       0.9))

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "messages":    messages,
            "max_tokens":  max_tokens,
            "temperature": temperature,
            "top_p":       top_p,
            "stream":      False,
        }

        try:
            # İlk istekte SYCL shader compilation olabilir (3-5 dakika)
            # Sonraki istekler çok daha hızlı
            resp = requests.post(
                f"{self._base_url}/v1/chat/completions",
                json=payload,
                timeout=600,
            )
            resp.raise_for_status()
            data = resp.json()

            msg   = data["choices"][0]["message"]
            usage = data.get("usage", {})
            text  = msg.get("content") or ""
            # Qwen3 thinking mode: content boş, reasoning_content'te asıl yanıt
            if not text.strip():
                text = msg.get("reasoning_content", "")
            # Hâlâ boşsa tüm string alanları birleştir
            if not text.strip():
                text = " ".join(str(v) for v in msg.values() if isinstance(v, str) and v.strip())
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            elapsed = time.time() - t0
            # completion_tokens 0 gelirse kelime sayısından tahmin et
            if completion_tokens == 0:
                completion_tokens = max(1, len(text.split()) * 4 // 3)
            tps = round(completion_tokens / elapsed, 2) if elapsed > 0 else 0

            metrics = {
                "backend":           "llama-server",
                "device":            self._device,
                "model":             self._model_name,
                "elapsed_s":         round(elapsed, 2),
                "tokens_per_second": tps,
                "prompt_tokens":     prompt_tokens,
                "completion_tokens": completion_tokens,
            }
            logger.info(
                f"LlamaServerBackend: {completion_tokens} token, "
                f"{tps} t/s, {round(elapsed,2)}s"
            )
            return text, metrics

        except requests.exceptions.ConnectionError:
            msg = f"llama-server bağlantı hatası: {self._base_url} — sunucu çalışıyor mu?"
            logger.error(msg)
            return "", {"error": msg}
        except Exception as e:
            logger.error(f"LlamaServerBackend generate hatası: {e}", exc_info=True)
            return "", {"error": str(e)}

    def generate_raw(self, prompt: str, params: dict) -> tuple[str, dict]:
        """Ham prompt ile /v1/completions endpoint'i."""
        t0 = time.time()
        payload = {
            "prompt":      prompt,
            "max_tokens":  int(params.get("max_tokens", 256)),
            "temperature": float(params.get("temperature", 0.7)),
            "stop":        params.get("stop", []),
        }
        try:
            resp = requests.post(
                f"{self._base_url}/v1/completions",
                json=payload, timeout=60,
            )
            resp.raise_for_status()
            data  = resp.json()
            text  = data["choices"][0]["text"]
            elapsed = time.time() - t0
            return text, {"elapsed_s": round(elapsed, 2), "backend": "llama-server"}
        except Exception as e:
            logger.error(f"LlamaServerBackend generate_raw hatası: {e}")
            return "", {"error": str(e)}

    def get_memory_info(self) -> dict:
        """llama-server props endpoint'inden bellek bilgisi."""
        try:
            r = requests.get(f"{self._base_url}/props", timeout=3)
            if r.status_code == 200:
                return {"ok": True, "result": r.json()}
        except Exception:
            pass
        return {"ok": False, "result": {}}
