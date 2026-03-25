"""
core/orchestrator.py
Tüm modülleri koordine eden ana sınıf.
Backend: OpenVINO | Ollama | LlamaCpp (GGUF/SYCL)

IPEX-LLM EOL Notu (Mart 2026):
  ipex-llm ve ipex_worker.py TCP mimarisi kaldırıldı.
  "ipex" backend artık LlamaCppBackend (GGUF) kullanır.
  Intel Arc iGPU hızlandırma için:
    a) Ollama backend + IPEX Ollama fork
    b) LlamaCpp backend + SYCL derlemeli llama-cpp-python
"""

import uuid
import logging
import threading
from typing import Optional, Generator
from pathlib import Path

from modules.database import DatabaseManager
from modules.model_manager import ModelScanner, ModelLoader, ModelInfo
from modules.search_engine import WebSearcher
from modules.dspy_enricher import DSPyEnricher
from modules.ipex_backend import OllamaBackend, LlamaCppBackend, IPEXModelInfo
from modules.ipex_worker_client import IPEXWorkerClient
from modules.hf_catalog import HFModelCatalog, CatalogEntry

logger = logging.getLogger(__name__)

JUNK_DOMAINS = {
    "sozluk.gov.tr", "nedemek.org", "nedirnedemek.com",
    "seslisozluk.net", "tureng.com", "bab.la", "reverso.net",
    "turkdiliveedebiyati.com", "nedir.com", "anlami.net",
}


class Orchestrator:
    BACKENDS = ("openvino", "ollama", "ipex")

    def __init__(self):
        self.session_id = str(uuid.uuid4())[:8]
        self._lock = threading.Lock()
        self._active_backend: str = "openvino"

        logger.info("Orchestrator başlatılıyor...")
        self.db      = DatabaseManager()
        self.scanner = ModelScanner()

        # OpenVINO backend
        self.ov_loader = ModelLoader(db_manager=self.db)

        # Ollama backend (standart + IPEX fork destekli)
        self.ollama = OllamaBackend(db_manager=self.db)

        # llama-cpp-python backend (GGUF / SYCL)
        # IPEXWorkerClient artık LlamaCppBackend'i sarmalar;
        # ayrı conda env veya TCP socket gerektirmez.
        self.ipex = IPEXWorkerClient(
            conda_env  = "openvino_studio",  # artık kullanılmıyor, uyumluluk
            port       = 62000,              # artık kullanılmıyor
            db_manager = self.db,
        )

        self.searcher = WebSearcher(db_manager=self.db)
        self.enricher = DSPyEnricher(db_manager=self.db)
        self.catalog  = HFModelCatalog()

        self._ov_models: list = []
        self._current_model_name: str = ""

        self.db.log_general(self.session_id, "INFO", "Orchestrator",
                            "Studio başlatıldı.", {"session": self.session_id})
        logger.info(f"Orchestrator hazır. Session: {self.session_id}")

    # ─────────────────────── Backend Yönetimi ────────────────────

    def set_backend(self, backend: str):
        with self._lock:
            if backend in self.BACKENDS:
                self._active_backend = backend
                logger.info(f"Backend değiştirildi: {backend}")

    def get_backend_status(self) -> dict:
        # Ollama durumu — IPEX fork mu standart mı?
        ollama_ok = self.ollama.is_available()
        if ollama_ok:
            btype = self.ollama.detect_backend_type()
            ollama_label = (
                "✅ IPEX Ollama fork (iGPU hızlandırma aktif)"
                if btype == "ipex_ollama"
                else "✅ Standart Ollama"
            )
        else:
            ollama_label = "❌ Çalışmıyor"

        # llama-cpp-python durumu
        lcpp = self.ipex._backend
        if lcpp.is_available():
            sycl = lcpp.check_sycl_support()
            lcpp_label = (
                f"✅ llama-cpp-python {'+ SYCL/XPU' if sycl else '(CPU only)'}"
            )
        else:
            lcpp_label = "❌ llama-cpp-python kurulu değil"

        return {
            "openvino": "✅ Mevcut",
            "ollama":   ollama_label,
            "ipex":     lcpp_label,
        }

    @property
    def _active_loader(self):
        if self._active_backend == "ollama":
            return self.ollama
        if self._active_backend == "ipex":
            return self.ipex
        return self.ov_loader

    # ─────────────────────── Model Tarama ────────────────────────

    def scan_models(self) -> list:
        with self._lock:
            backend = self._active_backend

        try:
            if backend == "openvino":
                ov_models = self.scanner.scan()
                with self._lock:
                    self._ov_models = ov_models
                self.db.log_general(self.session_id, "INFO", "Orchestrator",
                                    f"{len(ov_models)} OpenVINO modeli tarandı.")
                return ov_models

            elif backend == "ollama":
                models = self.ollama.list_models()
                self.db.log_general(self.session_id, "INFO", "Orchestrator",
                                    f"{len(models)} Ollama modeli listelendi.")
                return models

            elif backend == "ipex":
                return self._scan_ipex_local()

        except Exception as e:
            logger.error(f"Model tarama hatası: {e}")
            self.db.log_error(self.session_id, "Orchestrator.scan_models", e)
        return []

    def get_model_choices(self) -> list[str]:
        # Lock ekleyerek active_backend'in değişmesini engelliyoruz
        with self._lock:
            backend = self._active_backend
        
        # O anki backend ile scan yap (lock dışında, uzun sürebilir)
        try:
            if backend == "openvino":
                models = self.scanner.scan()
                with self._lock:
                    self._ov_models = models
            elif backend == "ollama":
                models = self.ollama.list_models()
            elif backend == "ipex":
                models = self._scan_ipex_local()
            else:
                models = []
        except Exception as e:
            logger.error(f"Model tarama hatası: {e}")
            models = []

        if backend == "openvino":
            if not models:
                return []
            return [f"{m.name} [{m.model_type.upper()}] ({m.size_mb:.0f} MB)"
                    for m in models]

        elif backend == "ollama":
            if not models:
                return []
            return [f"{m.model_id} ({m.size_gb:.1f} GB)" for m in models]

        elif backend == "ipex":
            if not models:
                return ["⚠️ GGUF model yok — Model Galerisi'nden indirin"]
            return [f"{m.name}  ({m.size_gb:.1f} GB)  —  {m.model_id}" for m in models]

        return []

    # ─────────────────────── Model Yükleme ───────────────────────

    def load_model(self, model_display_name: str, device: str = "CPU",
                   ov_config: dict = None) -> tuple[bool, str]:
        with self._lock:
            backend = self._active_backend

        try:
            if backend == "openvino":
                return self._load_openvino(model_display_name, device, ov_config=ov_config)
            elif backend == "ollama":
                return self._load_ollama(model_display_name)
            elif backend == "ipex":
                return self._load_ipex(model_display_name, device)
            return False, "Bilinmeyen backend."
        except Exception as e:
            self.db.log_error(self.session_id, "Orchestrator.load_model", e)
            return False, f"Model yükleme hatası: {e}"

    def _load_openvino(self, display_name: str, device: str,
                       ov_config: dict = None) -> tuple[bool, str]:
        if not self._ov_models:
            self._ov_models = self.scanner.scan()
        model_info = next((m for m in self._ov_models if m.name in display_name), None)
        if not model_info:
            return False, f"OpenVINO modeli bulunamadı: {display_name}"
        success, msg = self.ov_loader.load(
            model_info.path, device=device,
            session_id=self.session_id,
            ov_config=ov_config,
        )
        if success:
            with self._lock:
                self._current_model_name = model_info.name
            self.db.log_session(self.session_id, model_info.name, model_info.model_type)
            self.enricher.set_loader(self.ov_loader)
            self.enricher.configure_lm(model_info.path, device)
        return success, msg

    def _load_ollama(self, display_name: str) -> tuple[bool, str]:
        model_id = display_name.split(" (")[0].strip()
        success, msg = self.ollama.load(model_id, session_id=self.session_id)
        if success:
            with self._lock:
                self._current_model_name = f"ollama/{model_id}"
            self.db.log_session(self.session_id, model_id, "ollama")
            self.enricher.set_loader(self.ollama)
            self.enricher.configure_lm(model_id)
        return success, msg

    def _load_ipex(self, display_name: str, device: str) -> tuple[bool, str]:
        """
        GGUF model yükle.
        Format: "[GGUF] model-adı  (X.X GB)  —  /tam/yol/model.gguf"
        """
        if display_name.startswith("⚠️"):
            return False, "Önce Model Galerisi'nden veya HuggingFace'den bir .gguf modeli indirin."

        # Model yolunu çıkar
        if "—" in display_name:
            model_path = display_name.split("—")[-1].strip()
        else:
            model_path = display_name.strip()

        # Cihaz dönüşümü: UI "GPU"/"AUTO" → "xpu"
        ipex_device = "xpu" if device.upper() in ("GPU", "AUTO") else "cpu"

        success, msg = self.ipex.load(model_path, device=ipex_device,
                                      session_id=self.session_id)
        if success:
            with self._lock:
                self._current_model_name = f"llamacpp/{Path(model_path).stem}"
            self.db.log_session(self.session_id, Path(model_path).stem, "llamacpp")
            self.enricher.set_loader(self.ipex)
            self.enricher.configure_lm(model_path)
        return success, msg

    # ─────────────────────── Yerel Model Tarama ──────────────────

    def _scan_ipex_local(self) -> list:
        """
        Yerel GGUF dosyalarını tara (LlamaCppBackend aracılığıyla).
        C:\\OpenVINO_LLM\\gguf\\ ve diğer tanımlı dizinlerde arar.
        """
        return self.ipex.scan_local_gguf()

    # ─────────────────────── Katalog İşlemleri ───────────────────

    def get_openvino_catalog(self, search: str = "",
                             force_refresh: bool = False) -> list[CatalogEntry]:
        """HF'den canlı OpenVINO model listesi — yerel olanları işaretle."""
        try:
            entries = self.catalog.get_openvino_models(
                search=search, limit=80, force_refresh=force_refresh
            )
        except Exception as e:
            logger.error(f"OpenVINO katalog çekme hatası: {e}", exc_info=True)
            entries = []

        # Yerel tarama: C:\OpenVINO_LLM\ altındaki model klasörleri
        local_models = self.scanner.scan()
        tag_label    = "✅ İndirildi"

        for e in entries:
            model_name = e.model_id.split("/")[-1].lower()
            is_local   = any(model_name in m.path.lower() for m in local_models)
            if is_local:
                e.tags = [tag_label] + [t for t in e.tags if t != tag_label]
            else:
                e.tags = [t for t in e.tags if t != tag_label]
        return entries

    def get_ipex_catalog(self, search: str = "",
                         force_refresh: bool = False) -> list[CatalogEntry]:
        """
        GGUF model kataloğu.

        Yerel GGUF dosyalarını + HF'den GGUF model listesini döndürür.
        HF'deki GGUF modeller catalog.get_gguf_models() ile çekilir;
        yoksa fallback statik liste kullanılır.
        """
        try:
            entries = self.catalog.get_gguf_models(
                search=search, limit=60, force_refresh=force_refresh
            )
        except Exception as e:
            logger.error(f"GGUF katalog çekme hatası: {e}", exc_info=True)
            entries = []

        # Yerel GGUF dosyaları
        try:
            local_gguf = {m.model_id for m in self._scan_ipex_local()}
        except Exception as e:
            logger.error(f"Yerel GGUF taranırken hata: {e}")
            local_gguf = set()

        tag_label = "✅ İndirildi"
        for e in entries:
            is_local = (e.model_id in local_gguf or
                        Path(e.model_id).name in {Path(p).name for p in local_gguf})
            if is_local:
                e.tags = [tag_label] + [t for t in e.tags if t != tag_label]
            else:
                e.tags = [t for t in e.tags if t != tag_label]
        return entries

    def get_ollama_catalog(self, search: str = "",
                           force_refresh: bool = False) -> list[CatalogEntry]:
        """Ollama + HF'den birleşik katalog; yüklü modelleri işaretle."""
        try:
            entries = self.catalog.get_ollama_models(
                search=search, limit=60, force_refresh=force_refresh
            )
        except Exception as e:
            logger.error(f"Ollama katalog çekme hatası: {e}", exc_info=True)
            entries = []

        try:
            installed_ids = {m.model_id for m in self.ollama.list_models()}
        except Exception as e:
            logger.error(f"Ollama yerel modeller listelenirken hata: {e}")
            installed_ids = set()

        for e in entries:
            e.tags = (
                ["✅ Yüklü"] + [t for t in e.tags if t != "✅ Yüklü"]
                if e.model_id in installed_ids else
                [t for t in e.tags if t != "✅ Yüklü"]
            )
        return entries

    # ─────────────────────── İndirme İşlemleri ───────────────────

    def download_openvino_model(self, model_id: str,
                                dest_dir: str = r"C:\OpenVINO_LLM") -> tuple[bool, str]:
        """HF'den OpenVINO modelini belirtilen dizine indir."""
        try:
            from huggingface_hub import snapshot_download
            import os
            local_name = model_id.replace("/", "--")
            target = os.path.join(dest_dir, local_name)
            path = snapshot_download(repo_id=model_id, local_dir=target,
                                     local_files_only=False)
            return True, f"✅ İndirildi: {path}"
        except Exception as e:
            return False, f"❌ İndirme hatası: {e}"

    def download_gguf_model(self, model_id: str,
                            filename: str = "",
                            dest_dir: str = r"C:\OpenVINO_LLM\gguf") -> tuple[bool, str]:
        """
        HF'den GGUF dosyasını indir.

        model_id:  HF repo id (örn. "Qwen/Qwen2.5-7B-Instruct-GGUF")
        filename:  Repo içindeki .gguf dosya adı (boş bırakılırsa en küçük q4 dosyası seçilir)
        dest_dir:  Yerel hedef dizin
        """
        try:
            from huggingface_hub import hf_hub_download, list_repo_files
            import os

            os.makedirs(dest_dir, exist_ok=True)

            # Dosya adı belirtilmemişse akıllı seçim
            if not filename:
                all_files = list(list_repo_files(model_id))
                gguf_files = [f for f in all_files if f.endswith(".gguf")]
                if not gguf_files:
                    return False, f"'{model_id}' repo'sunda .gguf dosyası bulunamadı."

                # Tercih sırası: Q4_K_M > Q4_K_S > Q4_0 > diğerleri
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
            )
            return True, f"✅ İndirildi: {path}"

        except Exception as e:
            return False, f"❌ GGUF indirme hatası: {e}"

    def download_ipex_model(self, model_id: str) -> tuple[bool, str]:
        """Geriye uyumluluk — download_gguf_model'e yönlendirir."""
        # GGUF formatına yönlendir
        gguf_repo = model_id
        if not gguf_repo.endswith("-GGUF"):
            # Bilinen GGUF reposu adlandırma kurallarını dene
            org, name = (gguf_repo.split("/") + [""])[:2]
            possible_repos = [
                f"{org}/{name}-GGUF",
                f"bartowski/{name}-GGUF",
                f"TheBloke/{name}-GGUF",
            ]
        else:
            possible_repos = [gguf_repo]

        for repo in possible_repos:
            ok, msg = self.download_gguf_model(repo)
            if ok:
                return ok, msg

        # Son çare: klasik HF snapshot (büyük, önerilmez)
        return False, (
            f"'{model_id}' için GGUF repo bulunamadı.\n"
            "Lütfen HuggingFace'den doğru GGUF repo adını girin.\n"
            "Örnek: Qwen/Qwen2.5-7B-Instruct-GGUF"
        )

    def pull_ollama_model(self, model_id: str) -> tuple[bool, str]:
        return self.ollama.pull_model(model_id)

    def invalidate_catalog_cache(self, source: str = None):
        self.catalog.invalidate_cache(source)

    # ─────────────────────── Pipeline ────────────────────────────

    def run_pipeline(self, prompt: str, params: dict,
                     enable_search: bool = True,
                     enable_dspy: bool = True,
                     num_search_results: int = 5,
                     system_prompt: str = "") -> Generator[str, None, None]:

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
                        if r.url not in unique_urls and not any(d in r.url for d in JUNK_DOMAINS):
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
            if enable_dspy:
                yield "🧠 Prompt ve Görevler Yapılandırılıyor (DSPy)..."
                try:
                    enrichment = self.enricher.enrich(
                        prompt, search_context=search_context,
                        session_id=self.session_id
                    )
                    final_prompt = enrichment.enriched_prompt
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
                response, metrics = loader.generate(
                    prompt=final_prompt, params=params,
                    session_id=self.session_id, raw_prompt=prompt,
                    system_prompt=system_prompt,
                )
                logger.info(f"generate() tamamlandı — metrics={metrics}")

                if metrics.get("error"):
                    err = metrics["error"]
                    logger.error(f"Generate metrics error: {err}")
                    self.db.log_error(self.session_id, "Pipeline.generate_metric", Exception(err))
                    yield f"❌ Model hatası: {err}"
                else:
                    yield "---\n"
                    yield response
                    tps = metrics.get("tokens_per_second", "?")
                    dur = metrics.get("duration_ms", 0)
                    tok = metrics.get("output_tokens", "?")
                    try:
                        dur_fmt = f"{float(dur):.0f}ms"
                    except Exception:
                        dur_fmt = str(dur)
                    yield (f"\n\n---\n📊 *{tok} token | {tps} tok/s | "
                           f"{dur_fmt} | {self._active_backend}*")

            except Exception as e:
                logger.error(f"Generate hatası: {e}", exc_info=True)
                self.db.log_error(self.session_id, "Pipeline.generate", e)
                yield f"\n❌ Model hatası: {type(e).__name__}: {e}"

        except Exception as e:
            logger.error(f"Pipeline genel hata: {e}")
            self.db.log_error(self.session_id, "Pipeline.general", e,
                              {"prompt": prompt[:200]})
            yield f"\n❌ Beklenmeyen hata: {e}"

    # ─────────────────────── Yardımcılar ─────────────────────────

    def get_logs(self, session_only: bool = False) -> dict:
        return self.db.get_all_logs(
            session_id=self.session_id if session_only else None
        )

    def clear_logs(self, table: str = "all") -> bool:
        return self.db.clear_logs(table)

    def get_stats(self) -> dict:
        loader = self._active_loader
        stats  = self.db.get_stats()
        if hasattr(loader, "get_memory_info"):
            stats.update(loader.get_memory_info())
        stats["session_id"]     = self.session_id
        stats["active_backend"] = self._active_backend
        stats["model_loaded"]   = loader.loaded_model_name
        stats["backend_status"] = self.get_backend_status()
        return stats

    def new_session(self):
        with self._lock:
            self.session_id = str(uuid.uuid4())[:8]
        self.db.log_general(self.session_id, "INFO", "Orchestrator",
                            "Yeni session başlatıldı.")

    @property
    def is_model_loaded(self) -> bool:
        return self._active_loader.is_loaded
