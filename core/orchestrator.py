"""
core/orchestrator.py
====================
Tüm sistemin (Arama, DSPy, Modeller ve Tool'lar) yönetimini sağlayan merkez birim.

NOT: Bu sınıf artık stateless'tir. UI-bazlı state yönetimi için StateManager kullanın.
"""

import json
import logging
import threading
import time
from typing import Generator, Optional, Tuple

from pydantic import ValidationError

# -- Proje İçi Modüller --
from modules.search_engine import WebSearcher
from modules.dspy_enricher import DSPyEnricher
from modules.model_manager import ModelScanner, ModelLoader
from modules.ipex_worker_client import IPEXWorkerClient
from modules.ipex_backend import OllamaBackend
from modules.database import DatabaseManager
from modules.tools import ToolDispatcher
from core.constants import SearchConfig
from core.state_manager import get_state_manager, StateManager

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Stateless orchestrator.
    UI-bazlı state yönetimi StateManager ile yapılır.
    """
    
    def __init__(self, ui_id: str = "default"):
        self.db = DatabaseManager()
        self._lock = threading.RLock()
        self.ui_id = ui_id
        
        # State manager'dan UI state'ini al
        self._state_manager = get_state_manager()
        
        # Shared components (stateless)
        self.scanner = ModelScanner()

        # 3 Farklı Backend:
        self.ov_loader = ModelLoader(db_manager=self.db)
        self.ipex = IPEXWorkerClient(db_manager=self.db)
        self.ollama = OllamaBackend(db_manager=self.db)

        # Arama ve DSPy
        self.searcher = WebSearcher(db_manager=self.db)
        self.enricher = DSPyEnricher(db_manager=self.db)

        # Otonom araç (ReAct Tool) yöneticisi
        self.tool_dispatcher = ToolDispatcher()
    
    @property
    def session_id(self) -> str:
        """State manager'dan session_id al."""
        return self._state_manager.get_session_id(self.ui_id)
    
    @property
    def _active_backend(self) -> str:
        """State manager'dan aktif backend'i al."""
        return self._state_manager.get_backend(self.ui_id)
    
    def set_backend(self, backend: str):
        """UI için aktif backend'i ayarla."""
        self._state_manager.set_backend(self.ui_id, backend)
        logger.info(f"[{self.ui_id}] Backend değiştirildi: {backend}")

    def new_session(self):
        """Yeni session başlat."""
        self._state_manager.new_session(self.ui_id)
        self.searcher.clear_cache()
        logger.info(f"[{self.ui_id}] Yeni session başlatıldı: {self.session_id}")

    # -------------------------------------------------------------
    # Tool Parser (Ajan çıktılarını yorumlama)
    # -------------------------------------------------------------
    def _parse_action_from_response(self, response_text: str) -> tuple[Optional[str], Optional[str]]:
        """
        Model yanıtında Action ve Action Input kalıplarını arar.
        Örn:
        Action: transfer_money
        Action Input: {"from_account": "...", "to_account": "...", "amount": 100}
        """
        action = None
        action_input = None

        # 1. Action bul
        action_idx = response_text.find("Action:")
        if action_idx != -1:
            action_line = response_text[action_idx:].split('\n')[0]
            action = action_line.replace("Action:", "").strip()

        # 2. Action Input bul
        input_idx = response_text.find("Action Input:")
        if input_idx != -1:
            input_line = response_text[input_idx:].split('\n')[0]
            action_input = input_line.replace("Action Input:", "").strip()

        return action, action_input

    # -------------------------------------------------------------
    # Backend Yönetimi
    # -------------------------------------------------------------
    @property
    def _active_loader(self):
        backend = self._active_backend
        if backend == "ollama":
            return self.ollama
        if backend == "ipex":
            return self.ipex
        return self.ov_loader

    # -------------------------------------------------------------
    # Bilgi ve Durum
    # -------------------------------------------------------------
    @property
    def is_model_loaded(self) -> bool:
        return self._active_loader.is_loaded

    def get_model_choices(self) -> list:
        """Arayüzde gösterilecek model listesi (backend'e göre)."""
        backend = self._active_backend

        try:
            if backend == "openvino":
                # Model cache'i state manager'dan al
                cached_models = self._state_manager.get_model_cache(self.ui_id)
                if cached_models:
                    choices = [f"{m.name}  ({m.precision})" for m in cached_models]
                    if choices:
                        return choices
                
                # Cache yoksa tara
                self.scanner = ModelScanner()
                models = self.scanner.scan()
                self._state_manager.set_model_cache(self.ui_id, models)
                choices = [f"{m.name}  ({m.precision})" for m in models]
                if not choices:
                    choices = ["OpenVINO modeli bulunamadı"]
                return choices

            if backend == "ollama":
                try:
                    import requests
                    r = requests.get(f"{self.ollama.base_url}/api/tags", timeout=5)
                    if r.status_code == 200:
                        models = r.json().get("models", [])
                        return [m["name"] for m in models] if models else ["Ollama modeli bulunamadı"]
                except Exception as e:
                    logger.error(f"Ollama tag okuma hatası: {e}")
                return ["Ollama'ya bağlanılamadı"]

            if backend == "ipex":
                local_ggufs = self.ipex.scan_local_gguf()
                return local_ggufs if local_ggufs else ["Yerel GGUF bulunamadı"]

        except Exception as e:
            logger.error(f"get_model_choices hatası: {e}")
        return []

    def get_backend_status(self) -> dict:
        """Tüm backend'lerin durumunu döndür."""
        ov_loaded = self.ov_loader.is_loaded
        
        ipex_stat = self.ipex.get_status()
        ipex_avail = ipex_stat.get("result", {}).get("xpu_available", False)
        ipex_loaded = ipex_stat.get("result", {}).get("loaded", False)
        
        ollama_ok = False
        ollama_loaded = False
        try:
            import requests
            r = requests.get(f"{self.ollama.base_url}/api/tags", timeout=2)
            if r.status_code == 200:
                ollama_ok = True
                ollama_loaded = self.ollama.is_loaded
        except:
            pass
        
        return {
            "openvino": "Yüklü ✅" if ov_loaded else "Hazır ✅",
            "ipex": "Yüklü ✅" if ipex_loaded else ("Hazır ✅" if ipex_avail else "SYCL Yok ❌"),
            "ollama": "Yüklü ✅" if ollama_loaded else ("Hazır ✅" if ollama_ok else "Çevrimdışı ❌")
        }

    def get_system_status(self) -> dict:
        try:
            import psutil
            mem = psutil.virtual_memory()
            cpu = psutil.cpu_percent()
        except:
            mem = None
            cpu = 0
        
        active_loader = self._active_loader
        return {
            "model_loaded": active_loader.is_loaded,
            "model_name": active_loader.loaded_model_name if active_loader.is_loaded else "",
            "backend": self._active_backend,
            "ram_percent": mem.percent if mem else 0,
            "cpu_percent": cpu
        }

    # -------------------------------------------------------------
    # Yükleme (Load)
    # -------------------------------------------------------------
    def load_model(self, display_name: str, device: str = "CPU",
                   ov_config: dict = None) -> tuple[bool, str]:
        backend = self._active_backend

        if backend == "openvino":
            return self._load_ov(display_name, device, ov_config)
        elif backend == "ollama":
            return self._load_ollama(display_name)
        elif backend == "ipex":
            return self._load_ipex(display_name, device)
        return False, "Geçersiz backend"

    def _load_ov(self, display_name: str, device: str = "CPU",
                       ov_config: dict = None) -> tuple[bool, str]:
        # Model cache'i state manager'dan al
        cached_models = self._state_manager.get_model_cache(self.ui_id)
        if not cached_models:
            self.scanner = ModelScanner()
            cached_models = self.scanner.scan()
            self._state_manager.set_model_cache(self.ui_id, cached_models)
        
        model_info = next((m for m in cached_models if display_name.startswith(m.name + " [")), None)
        if not model_info:
            return False, f"OpenVINO modeli bulunamadı: {display_name}"
        success, msg = self.ov_loader.load(
            model_info.path,
            device=device,
            config=ov_config
        )
        if success:
            self.enricher.set_loader(self.ov_loader)
        return success, msg

    def _load_ollama(self, model_name: str) -> tuple[bool, str]:
        try:
            import requests
            r = requests.post(
                f"{self.ollama.base_url}/api/generate",
                json={"model": model_name, "keep_alive": -1},
                timeout=30
            )
            if r.status_code == 200:
                self.ollama._current_model = model_name
                self.enricher.set_loader(self.ollama)
                return True, f"Ollama: '{model_name}' bellekte tutuluyor."
        except Exception as e:
            return False, f"Ollama yükleme hatası: {e}"
        return False, "Bilinmeyen Ollama hatası."

    def _load_ipex(self, model_path: str, device: str) -> tuple[bool, str]:
        # 'AUTO' cihazını algıla
        if device.upper() == "AUTO":
            device = "xpu" if self.ipex._backend.check_sycl_support() else "cpu"
        elif device.upper() == "GPU":
            device = "xpu"
            
        success, msg = self.ipex.load(
            model_id=model_path,
            device=device,
            session_id=self.session_id,
            n_ctx=4096
        )
        if success:
            self.enricher.set_loader(self.ipex)
        return success, msg

    # -------------------------------------------------------------
    # Veritabanı
    # -------------------------------------------------------------
    def get_logs(self, session_only: bool = True) -> dict:
        sess = self.session_id if session_only else None
        return {
            "search": self.db.get_search_logs(sess),
            "dspy": self.db.get_dspy_logs(sess),
            "llm": self.db.get_llm_logs(sess),
            "errors": self.db.get_error_logs(sess),
            "general": self.db.get_general_logs(sess),
        }

    def clear_logs(self, log_type: str = "all") -> bool:
        return self.db.clear_logs(log_type)

    # -------------------------------------------------------------
    # Model Katalogları
    # -------------------------------------------------------------
    def get_openvino_catalog(self, search: str = "", force: bool = False) -> list:
        """HuggingFace'den OpenVINO model kataloğunu döndürür."""
        from modules.hf_catalog import HFModelCatalog
        catalog = HFModelCatalog()
        if force:
            catalog.invalidate_cache("openvino")
        return catalog.get_openvino_models(search=search)

    def get_ollama_catalog(self, search: str = "", force: bool = False) -> list:
        """Ollama model kataloğunu döndürür (HF + fallback)."""
        from modules.hf_catalog import HFModelCatalog
        catalog = HFModelCatalog()
        if force:
            catalog.invalidate_cache("ollama")
        return catalog.get_ollama_models(search=search)

    def get_ipex_catalog(self, search: str = "", force: bool = False) -> list:
        """GGUF model kataloğunu döndürür (llama-cpp-python için)."""
        from modules.hf_catalog import HFModelCatalog
        catalog = HFModelCatalog()
        if force:
            catalog.invalidate_cache("gguf")
        return catalog.get_gguf_models(search=search)

    def get_hf_gguf_catalog(self) -> list:
        """HuggingFace'den GGUF modelleri katalogunu döndürür (legacy)."""
        from modules.hf_catalog import HFModelCatalog
        catalog = HFModelCatalog()
        return catalog.get_gguf_models()

    # -------------------------------------------------------------
    # İstatistikler ve Durum
    # -------------------------------------------------------------
    def get_stats(self) -> dict:
        """Sistem istatistikleri ve durum bilgilerini döndürür."""
        import psutil
        
        mem = psutil.virtual_memory()
        backend_status = self.get_backend_status()
        
        return {
            "session_id": self.session_id,
            "active_backend": self._active_backend,
            "model_loaded": self.is_model_loaded,
            "backend_status": backend_status,
            "total_gb": round(mem.total / 1024**3, 1),
            "used_gb": round(mem.used / 1024**3, 1),
            "available_gb": round(mem.available / 1024**3, 1),
            "percent": mem.percent,
            "total_searches": self.db.get_stats().get("total_searches", 0),
            "total_llm_calls": self.db.get_stats().get("total_llm_calls", 0),
            "total_dspy_calls": self.db.get_stats().get("total_dspy_calls", 0),
            "total_errors": self.db.get_stats().get("total_errors", 0),
            "db_size_mb": self.db.get_stats().get("db_size_mb", 0),
        }

    def pull_ollama_model(self, model_id: str) -> tuple[bool, str]:
        """Ollama'ya model indirir."""
        return self.ollama.pull_model(model_id)

    def download_openvino_model(self, model_id: str, dest_dir: str) -> tuple[bool, str]:
        """HuggingFace'den OpenVINO model indirir."""
        try:
            from huggingface_hub import snapshot_download
            import os
            
            os.makedirs(dest_dir, exist_ok=True)
            logger.info(f"OpenVINO modeli indiriliyor: {model_id} → {dest_dir}")
            
            path = snapshot_download(
                repo_id=model_id,
                local_dir=dest_dir,
                local_dir_use_symlinks=False,
            )
            return True, f"İndirme tamamlandı: {path}"
        except Exception as e:
            logger.error(f"OpenVINO indirme hatası: {e}")
            return False, f"Hata: {e}"

    def download_gguf_model(self, model_id: str,
                            filename: str = "",
                            dest_dir: str = r"C:\OpenVINO_LLM\gguf",
                            quant_hint: str = "") -> tuple[bool, str]:
        """
        HF'den GGUF dosyasını indir.

        model_id:   HF repo id (örn. "Qwen/Qwen2.5-7B-Instruct-GGUF")
        filename:   Repo içindeki .gguf dosya adı (boş bırakılırsa otomatik seçilir)
        dest_dir:   Yerel hedef dizin
        quant_hint: Tercih edilen quant türü, örn. "Q8_0", "Q4_K_M" (boş = otomatik)
        """
        try:
            from huggingface_hub import hf_hub_download, list_repo_files
            import os
            
            os.makedirs(dest_dir, exist_ok=True)
            logger.info(f"GGUF aranıyor: {model_id}...")
            
            if not filename:
                files = list_repo_files(model_id)
                gguf_files = [f for f in files if f.endswith(".gguf")]
                if not gguf_files:
                    return False, f"'{model_id}' repo'sunda .gguf dosyası bulunamadı."

                # quant_hint verilmişse önce onu dene
                if quant_hint and quant_hint.lower() != "otomatik":
                    hint_lower = quant_hint.lower()
                    hint_match = next((f for f in gguf_files if hint_lower in f.lower()), None)
                    if hint_match:
                        filename = hint_match
                        logger.info(f"GGUF quant hint uygulandı: {filename}")

                # hint yoksa veya bulunamadıysa varsayılan tercih sırası
                if not filename:
                    preferred = ["q4_k_m", "q4_k_s", "q4_0", "q5_k_m", "q8_0"]
                    selected  = None
                    for pref in preferred:
                        match = next((f for f in gguf_files if pref in f.lower()), None)
                        if match:
                            selected = match
                            break
                    filename = selected or gguf_files[0]
                logger.info(f"GGUF dosyası seçildi: {filename}")

            path = hf_hub_download(
                repo_id=model_id,
                filename=filename,
                local_dir=dest_dir,
                local_dir_use_symlinks=False
            )
            return True, f"İndirme tamamlandı: {path}"
        except Exception as e:
            logger.error(f"GGUF indirme hatası: {e}")
            return False, f"Hata: {e}"

    # -------------------------------------------------------------
    # Yürütme (Streaming Pipeline)
    # -------------------------------------------------------------
    def run_pipeline(self, prompt: str, params: dict,
                     enable_search: bool = True,
                     enable_dspy: bool = True,
                     num_search_results: int = 5,
                     system_prompt: str = "",
                     history: list = None) -> Generator[str, None, None]:

        loader = self._active_loader
        if not loader.is_loaded:
            yield "❌ Model yüklü değil. Lütfen önce bir model seçin ve yükleyin."
            return

        try:
            search_context = ""
            if enable_search:
                queries = []
                
                # 1. Sorgu Optimizasyonu (DSPy/LLM ile)
                if enable_dspy and loader.is_loaded:
                    yield "🧠 Arama sorguları oluşturuluyor (LLM)...\n"
                    try:
                        queries = self.enricher.generate_search_query(prompt)
                        yield f"✨ {len(queries)} farklı arama sorgusu planlandı: {', '.join(queries)}\n"
                    except Exception as e:
                        logger.warning(f"Arama sorgusu LLM hatası, fallback kullanılıyor: {e}")
                        queries = [prompt]
                else:
                    queries = [self.searcher.optimizer.extract_query(prompt)]

                # 2. Arama Çalıştırma (Çoklu Arama)
                all_results = []
                for idx, q in enumerate(queries):
                    yield f"🔍 Web araması [{idx+1}/{len(queries)}]: '{q}'...\n"
                    try:
                        results, _ = self.searcher.search(
                            prompt=q,
                            num_results=num_search_results,
                            session_id=self.session_id,
                            optimize_query=False  # Zaten optimize ettik veya optimize edilecek bir şey yok
                        )
                        if results:
                            all_results.extend(results)
                    except Exception as e:
                        logger.error(f"Arama hatası: {e}")
                        self.db.log_error(self.session_id, "Pipeline.search", e)
                        yield f"⚠️ '{q}' için arama hatası: {e}\n"

                # 3. Sonuçları Birleştir ve Temizle
                if all_results:
                    # Aynı URL'leri deduplicate et ve JUNK_DOMAINS filtresi uygula
                    unique_urls = set()
                    final_results = []
                    
                    # Sonuçları ilgi puanına (relevance_score) göre sıralayıp filtrele
                    for r in sorted(all_results, key=lambda x: x.relevance_score, reverse=True):
                        if r.url not in unique_urls and not any(d in r.url for d in SearchConfig.JUNK_DOMAINS):
                            unique_urls.add(r.url)
                            final_results.append(r)
                            
                    # En iyi num_search_results * 1.5 kadarını tut (çoklu soru olduğu için daha fazla bilgi gerekebilir)
                    final_results = final_results[:int(num_search_results * 1.5)]
                            
                    if final_results:
                        search_context = self.searcher.format_context(final_results)
                        yield f"✅ Toplam {len(final_results)} farklı ve kaliteli arama sonucu bağlama eklendi.\n"
                    else:
                        yield "⚠️ Arama sonuçları konu dışı veya kalitesiz, bağlam kullanılmıyor.\n"
                else:
                    yield "⚠️ Hiçbir arama sonucu bulunamadı.\n"

            final_prompt = prompt
            is_react_mode = False
            
            if enable_dspy:
                yield "🧠 Prompt ve Görevler Yapılandırılıyor (DSPy)..."
                try:
                    enrichment = self.enricher.enrich(
                        prompt, search_context=search_context,
                        session_id=self.session_id
                    )
                    final_prompt = enrichment.enriched_prompt
                    is_react_mode = (enrichment.mode == "ReAct")
                    yield f"✅ DSPy modu: **{enrichment.mode}** ({enrichment.mode_reason})\n"
                except Exception as e:
                    logger.error(f"DSPy hatası: {e}")
                    self.db.log_error(self.session_id, "Pipeline.dspy", e)
                    yield f"⚠️ DSPy hatası: {e}\n"
                    if search_context:
                        final_prompt = f"{prompt}\n\n{search_context}"
            elif search_context:
                final_prompt = f"{prompt}\n\n{search_context}"

            backend_label = {
                "openvino": "⚡ OpenVINO",
                "ollama":   "🦙 Ollama",
                "ipex":     "🔷 llama-cpp-python",
            }.get(self._active_backend, "⚡")
            yield f"{backend_label} yanıt üretiyor...\n\n"

            # Güvenlik: loader hâlâ yüklü mü?
            if not loader.is_loaded:
                yield "❌ Model DSPy aşamasında boşaltıldı. Lütfen yeniden yükleyin."
                return

            try:
                logger.info(
                    f"generate() çağrılıyor — backend={self._active_backend}, "
                    f"prompt_len={len(final_prompt)}, params={params}"
                )

                # Ollama için non-streaming generate kullan (streaming sorunlu)
                if self._active_backend == "ollama":
                    response, metrics = loader.generate(
                        prompt=final_prompt,
                        params=params,
                        session_id=self.session_id,
                        system_prompt=system_prompt,
                        history=history,
                    )
                    if response:
                        yield response
                    else:
                        yield f"❌ Hata: {metrics.get('error', 'Bilinmeyen hata')}"
                else:
                    # OpenVINO ve IPEX için streaming
                    response = ""
                    for token in loader.generate_stream(
                        prompt=final_prompt, params=params,
                        system_prompt=system_prompt,
                        history=history,
                    ):
                        response += token
                        yield token

                yield "\n---\n"

            except Exception as e:
                logger.error(f"Üretim hatası: {e}")
                self.db.log_error(self.session_id, "Pipeline.generate", e)
                yield f"❌ Hata: {e}"

        except Exception as e:
            logger.error(f"Pipeline ana hatası: {e}")
            self.db.log_error(self.session_id, "Pipeline.main", e)
            yield f"❌ Beklenmeyen Hata: {e}"
