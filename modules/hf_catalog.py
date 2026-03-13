"""
modules/hf_catalog.py

HuggingFace Hub'dan canlı model kataloğu çeker.
Dört kaynak:
  1. OpenVINO   — library="openvino" + task="text-generation"
  2. GGUF        — GGUF formatında modeller (llama-cpp-python ile çalışır)
                   library="gguf" veya dosya adı .gguf içerenler
  3. Ollama      — Ollama REST API + statik fallback listesi
  4. (Eski IPEX) — get_ipex_models() artık get_gguf_models()'a yönlendirir

Sonuçlar önbelleğe (TTL=30 dk) alınır; ağ yoksa statik fallback döner.
"""

import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# Veri sınıfı
# ─────────────────────────────────────────

@dataclass
class CatalogEntry:
    model_id: str          # HF repo id veya Ollama model adı
    name: str              # Görünen ad
    source: str            # "openvino" | "gguf" | "ollama"
    size_str: str = ""     # "4.7 GB" gibi tahmini boyut
    context: str = ""      # "32K", "128K" vb.
    tags: list = field(default_factory=list)
    downloads: int = 0
    likes: int = 0
    last_modified: str = ""
    gated: bool = False


# ─────────────────────────────────────────
# Önbellekli HF Catalog Çekicisi
# ─────────────────────────────────────────

class HFModelCatalog:
    """
    HuggingFace Hub'dan canlı model listesi çeker.
    Sonuçlar TTL=1800 sn (30 dk) önbelleğe alınır.
    """

    CACHE_TTL = 1800  # saniye

    # ── Statik Fallback Listeleri ──────────────────────────────────

    _FALLBACK_OV = [
        CatalogEntry("OpenVINO/Phi-3.5-mini-instruct-int4-ov",        "Phi-3.5-mini-instruct (INT4)",        "openvino", "~1.8 GB", "128K", ["hafif", "hızlı", "Microsoft"]),
        CatalogEntry("OpenVINO/phi-2-int4-ov",                         "Phi-2 (INT4)",                        "openvino", "~1.5 GB", "2K",   ["hafif", "Microsoft"]),
        CatalogEntry("OpenVINO/mistral-7b-instruct-v0.1-int4-ov",     "Mistral-7B-Instruct v0.1 (INT4)",     "openvino", "~3.8 GB", "32K",  ["stabil", "genel"]),
        CatalogEntry("OpenVINO/mistral-7b-instruct-v0.3-int8-ov",     "Mistral-7B-Instruct v0.3 (INT8)",     "openvino", "~7.2 GB", "32K",  ["yüksek kalite"]),
        CatalogEntry("OpenVINO/Llama-3.1-8B-Instruct-int4-ov",        "LLaMA-3.1-8B-Instruct (INT4)",        "openvino", "~4.6 GB", "128K", ["Meta", "uzun context"]),
        CatalogEntry("OpenVINO/Llama-3.2-3B-Instruct-int4-ov",        "LLaMA-3.2-3B-Instruct (INT4)",        "openvino", "~1.8 GB", "128K", ["hafif", "hızlı"]),
        CatalogEntry("OpenVINO/qwen2.5-7b-instruct-int4-ov",          "Qwen2.5-7B-Instruct (INT4)",          "openvino", "~4.3 GB", "32K",  ["Türkçe güçlü", "önerilen"]),
        CatalogEntry("OpenVINO/qwen2.5-14b-instruct-int4-ov",         "Qwen2.5-14B-Instruct (INT4)",         "openvino", "~8.5 GB", "32K",  ["büyük", "güçlü"]),
        CatalogEntry("OpenVINO/gemma-2-2b-it-int4-ov",                "Gemma-2-2B-IT (INT4)",                "openvino", "~1.4 GB", "8K",   ["hafif", "Google"]),
        CatalogEntry("OpenVINO/gemma-2-9b-it-int4-ov",                "Gemma-2-9B-IT (INT4)",                "openvino", "~5.2 GB", "8K",   ["Google", "kaliteli"]),
        CatalogEntry("OpenVINO/DeepSeek-R1-Distill-Qwen-7B-int4-ov", "DeepSeek-R1-Distill-Qwen-7B (INT4)", "openvino", "~4.5 GB", "32K",  ["akıl yürütme", "matematik"]),
    ]

    # GGUF fallback — bartowski, TheBloke ve resmi GGUF repoları
    _FALLBACK_GGUF = [
        CatalogEntry("Qwen/Qwen2.5-7B-Instruct-GGUF",                  "Qwen2.5-7B-Instruct (GGUF)",         "gguf", "~4.3 GB", "32K",  ["⭐ önerilen", "Türkçe güçlü", "Qwen"]),
        CatalogEntry("Qwen/Qwen2.5-14B-Instruct-GGUF",                 "Qwen2.5-14B-Instruct (GGUF)",        "gguf", "~8.5 GB", "32K",  ["büyük", "güçlü", "Qwen"]),
        CatalogEntry("Qwen/Qwen3-8B-GGUF",                             "Qwen3-8B (GGUF)",                    "gguf", "~5.0 GB", "32K",  ["⭐ önerilen", "en yeni", "Qwen"]),
        CatalogEntry("bartowski/Mistral-7B-Instruct-v0.3-GGUF",       "Mistral-7B-Instruct v0.3 (GGUF)",    "gguf", "~4.1 GB", "32K",  ["hızlı", "stabil"]),
        CatalogEntry("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",     "LLaMA-3.1-8B-Instruct (GGUF)",       "gguf", "~4.7 GB", "128K", ["Meta", "uzun context"]),
        CatalogEntry("bartowski/Meta-Llama-3.2-3B-Instruct-GGUF",     "LLaMA-3.2-3B-Instruct (GGUF)",       "gguf", "~2.0 GB", "128K", ["hafif", "hızlı"]),
        CatalogEntry("bartowski/Phi-3.5-mini-instruct-GGUF",          "Phi-3.5-mini-instruct (GGUF)",        "gguf", "~2.2 GB", "128K", ["hafif", "uzun context", "Microsoft"]),
        CatalogEntry("bartowski/gemma-2-9b-it-GGUF",                  "Gemma-2-9B-IT (GGUF)",                "gguf", "~5.5 GB", "8K",   ["Google", "kaliteli"]),
        CatalogEntry("bartowski/gemma-2-2b-it-GGUF",                  "Gemma-2-2B-IT (GGUF)",                "gguf", "~1.5 GB", "8K",   ["hafif", "Google"]),
        CatalogEntry("bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF",   "DeepSeek-R1-Distill-Qwen-7B (GGUF)", "gguf", "~4.5 GB", "32K",  ["akıl yürütme", "matematik"]),
        CatalogEntry("bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF",  "DeepSeek-R1-Distill-Llama-8B (GGUF)", "gguf", "~5.0 GB", "32K", ["akıl yürütme", "güçlü"]),
        CatalogEntry("microsoft/Phi-4-GGUF",                           "Phi-4 (GGUF)",                        "gguf", "~8.5 GB", "16K",  ["Microsoft", "kod"]),
        CatalogEntry("bartowski/phi-4-GGUF",                           "Phi-4 (GGUF, bartowski)",             "gguf", "~8.5 GB", "16K",  ["Microsoft", "kod"]),
        CatalogEntry("unsloth/Qwen3-8B-GGUF",                         "Qwen3-8B (GGUF, unsloth)",            "gguf", "~5.0 GB", "32K",  ["en yeni", "optimize"]),
    ]

    _FALLBACK_OLLAMA = [
        CatalogEntry("qwen2.5:7b",       "Qwen2.5 7B",       "ollama", "4.7 GB", "32K",  ["Türkçe güçlü", "önerilen"]),
        CatalogEntry("qwen2.5:14b",      "Qwen2.5 14B",      "ollama", "9.0 GB", "32K",  ["büyük", "güçlü"]),
        CatalogEntry("qwen3:8b",         "Qwen3 8B",         "ollama", "5.2 GB", "32K",  ["en yeni", "düşünme", "önerilen"]),
        CatalogEntry("qwen3:14b",        "Qwen3 14B",        "ollama", "9.3 GB", "32K",  ["en yeni", "büyük"]),
        CatalogEntry("mistral:7b",       "Mistral 7B",       "ollama", "4.1 GB", "32K",  ["hızlı", "stabil"]),
        CatalogEntry("mistral-nemo:12b", "Mistral Nemo 12B", "ollama", "7.1 GB", "128K", ["uzun context"]),
        CatalogEntry("phi4:14b",         "Phi-4 14B",        "ollama", "8.9 GB", "16K",  ["Microsoft", "kod"]),
        CatalogEntry("phi3.5:3.8b",      "Phi-3.5 3.8B",     "ollama", "2.2 GB", "128K", ["hafif", "uzun context"]),
        CatalogEntry("gemma3:9b",        "Gemma 3 9B",       "ollama", "5.4 GB", "8K",   ["Google", "çok dilli"]),
        CatalogEntry("gemma3:4b",        "Gemma 3 4B",       "ollama", "3.3 GB", "8K",   ["Google", "hafif"]),
        CatalogEntry("llama3.1:8b",      "LLaMA 3.1 8B",     "ollama", "4.7 GB", "128K", ["Meta", "önerilen"]),
        CatalogEntry("llama3.2:3b",      "LLaMA 3.2 3B",     "ollama", "2.0 GB", "128K", ["hafif", "hızlı"]),
        CatalogEntry("deepseek-r1:7b",   "DeepSeek-R1 7B",   "ollama", "4.7 GB", "32K",  ["akıl yürütme", "matematik"]),
        CatalogEntry("deepseek-r1:14b",  "DeepSeek-R1 14B",  "ollama", "9.0 GB", "32K",  ["akıl yürütme", "büyük"]),
        CatalogEntry("falcon3:7b",       "Falcon 3 7B",      "ollama", "4.7 GB", "32K",  ["TII", "çok dilli"]),
        CatalogEntry("command-r7b:7b",   "Command-R 7B",     "ollama", "4.7 GB", "128K", ["RAG odaklı", "Cohere"]),
        CatalogEntry("internlm3:8b",     "Internlm3 8B",     "ollama", "5.0 GB", "32K",  ["Çince/İngilizce"]),
        CatalogEntry("minicpm3:4b",      "MiniCPM3 4B",      "ollama", "2.5 GB", "32K",  ["hafif", "etkileyici"]),
    ]

    def __init__(self):
        self._cache: dict[str, tuple[float, list[CatalogEntry]]] = {}
        self._lock = threading.Lock()

    # ─── Ana Arayüz ───────────────────────────────────────────────

    def get_openvino_models(self, search: str = "", limit: int = 60,
                            force_refresh: bool = False) -> list[CatalogEntry]:
        return self._get("openvino", search, limit, force_refresh)

    def get_gguf_models(self, search: str = "", limit: int = 60,
                        force_refresh: bool = False) -> list[CatalogEntry]:
        """GGUF formatında model listesi (llama-cpp-python ile kullanılır)."""
        return self._get("gguf", search, limit, force_refresh)

    def get_ipex_models(self, search: str = "", limit: int = 60,
                        force_refresh: bool = False) -> list[CatalogEntry]:
        """
        Geriye uyumluluk — get_gguf_models'a yönlendirir.
        ipex-llm EOL olduğundan GGUF listesi döndürülür.
        """
        logger.debug("get_ipex_models() → get_gguf_models() yönlendiriliyor (ipex-llm EOL)")
        return self.get_gguf_models(search=search, limit=limit,
                                    force_refresh=force_refresh)

    def get_ollama_models(self, search: str = "", limit: int = 60,
                          force_refresh: bool = False) -> list[CatalogEntry]:
        return self._get("ollama", search, limit, force_refresh)

    # ─── Önbellekli Çekici ────────────────────────────────────────

    def _get(self, source: str, search: str, limit: int,
             force_refresh: bool) -> list[CatalogEntry]:
        cache_key = f"{source}:{search}:{limit}"
        with self._lock:
            if not force_refresh and cache_key in self._cache:
                ts, data = self._cache[cache_key]
                if time.time() - ts < self.CACHE_TTL:
                    logger.debug(f"Katalog önbellekten döndü: {cache_key}")
                    return data

        try:
            if source == "openvino":
                data = self._fetch_openvino(search, limit)
            elif source == "gguf":
                data = self._fetch_gguf(search, limit)
            elif source == "ollama":
                data = self._fetch_ollama_hf(search, limit)
            else:
                data = []

            if data:
                with self._lock:
                    self._cache[cache_key] = (time.time(), data)
                return data
        except Exception as e:
            logger.warning(f"HF katalog çekme hatası ({source}): {e} — fallback kullanılıyor")

        # Fallback
        fallback = {
            "openvino": self._FALLBACK_OV,
            "gguf":     self._FALLBACK_GGUF,
            "ollama":   self._FALLBACK_OLLAMA,
        }.get(source, [])

        if search:
            sl = search.lower()
            fallback = [
                e for e in fallback
                if sl in e.model_id.lower() or sl in e.name.lower()
                or any(sl in t.lower() for t in e.tags)
            ]
        return fallback

    # ─── OpenVINO Çekici ──────────────────────────────────────────

    def _fetch_openvino(self, search: str, limit: int) -> list[CatalogEntry]:
        from huggingface_hub import HfApi
        api = HfApi()

        kwargs = dict(
            task="text-generation",
            library="openvino",
            sort="downloads",
            direction=-1,
            limit=limit,
            full=False,
            cardData=False,
        )
        if search:
            kwargs["search"] = search

        entries = []
        try:
            for m in api.list_models(**kwargs):
                tags = list(m.tags or [])
                if any(t in tags for t in ["vision", "image", "audio", "speech"]):
                    continue
                entries.append(self._hf_to_entry(m, "openvino"))
        except Exception as e:
            logger.warning(f"OpenVINO HF sorgusu başarısız: {e}")
            raise

        entries.sort(key=lambda e: (0 if e.model_id.startswith("OpenVINO/") else 1,
                                    -e.downloads))
        return entries

    # ─── GGUF Çekici ──────────────────────────────────────────────

    def _fetch_gguf(self, search: str, limit: int) -> list[CatalogEntry]:
        """
        HF'de GGUF formatında model arama.
        library="gguf" filtresi kullanılır.
        Bilinen kaliteli GGUF sağlayıcılar önceliklendirilir.
        """
        from huggingface_hub import HfApi
        api = HfApi()

        PRIORITY_AUTHORS = {
            "bartowski", "Qwen", "unsloth", "TheBloke",
            "microsoft", "google", "meta-llama",
        }

        kwargs = dict(
            library="gguf",
            sort="downloads",
            direction=-1,
            limit=limit * 2,
            full=False,
            cardData=False,
        )
        if search:
            kwargs["search"] = search

        entries = []
        seen    = set()

        try:
            for m in api.list_models(**kwargs):
                mid = m.modelId if hasattr(m, "modelId") else m.id
                if mid in seen:
                    continue

                # Yalnızca text-generation / LLM
                tags = list(m.tags or [])
                if any(t in tags for t in ["vision", "image-text", "audio",
                                            "speech", "embedding", "reranker"]):
                    continue

                entry  = self._hf_to_entry(m, "gguf")
                author = mid.split("/")[0]

                if author in PRIORITY_AUTHORS:
                    entry.tags = ["⭐ önerilen"] + entry.tags

                entries.append(entry)
                seen.add(mid)

                if len(entries) >= limit:
                    break

        except Exception as e:
            logger.warning(f"GGUF HF sorgusu başarısız: {e}")
            raise

        return entries

    # ─── Ollama HF Çekici ─────────────────────────────────────────

    def _fetch_ollama_hf(self, search: str, limit: int) -> list[CatalogEntry]:
        from huggingface_hub import HfApi
        api = HfApi()

        OLLAMA_FAMILIES = ["Qwen", "mistral", "llama", "gemma", "phi",
                           "deepseek", "falcon", "command-r"]

        hf_entries = []
        for family in OLLAMA_FAMILIES:
            try:
                query  = f"{family} {search}".strip() if search else family
                models = api.list_models(
                    task="text-generation",
                    search=query,
                    sort="downloads",
                    direction=-1,
                    limit=5,
                    gated=False,
                    full=False,
                    cardData=False,
                )
                for m in list(models)[:3]:
                    mid = m.modelId if hasattr(m, "modelId") else m.id
                    if any(k in mid.lower() for k in ["instruct", "chat", "it"]):
                        hf_entries.append(self._hf_to_entry(m, "ollama_hf"))
            except Exception:
                pass

        combined  = list(self._FALLBACK_OLLAMA)
        seen_ids  = {e.model_id for e in combined}

        if search:
            sl       = search.lower()
            combined = [
                e for e in combined
                if sl in e.model_id.lower() or sl in e.name.lower()
                or any(sl in t.lower() for t in e.tags)
            ]

        for e in hf_entries:
            base = e.model_id.split("/")[-1].lower()
            if not any(base in s for s in seen_ids):
                combined.append(e)

        return combined[:limit]

    # ─── Yardımcılar ──────────────────────────────────────────────

    def _hf_to_entry(self, m, source: str) -> CatalogEntry:
        mid       = m.modelId if hasattr(m, "modelId") else m.id
        tags      = list(m.tags or [])
        downloads = getattr(m, "downloads", 0) or 0
        likes     = getattr(m, "likes", 0) or 0
        last_mod  = str(getattr(m, "lastModified", "") or "")[:10]
        gated     = getattr(m, "gated", False) or False
        display   = mid.split("/")[-1]

        size_str  = self._estimate_size_from_name(display)
        ctx       = self._estimate_context(tags, display)

        useful_tags = []
        for t in tags:
            tl = t.lower()
            if any(k in tl for k in ["license", "dataset", "arxiv", "base_model",
                                      "transformers", "pytorch", "safetensors",
                                      "openvino", "gguf", "bnb", "auto"]):
                continue
            if len(t) < 3 or len(t) > 20:
                continue
            useful_tags.append(t)

        return CatalogEntry(
            model_id=mid,
            name=display,
            source=source,
            size_str=size_str,
            context=ctx,
            tags=useful_tags[:5],
            downloads=downloads,
            likes=likes,
            last_modified=last_mod,
            gated=gated,
        )

    def _estimate_size_from_name(self, name: str) -> str:
        nl = name.lower()
        for pattern, size in [
            ("0.5b", "~0.4 GB"), ("1b",   "~0.7 GB"), ("1.5b", "~1.0 GB"),
            ("2b",   "~1.3 GB"), ("3b",   "~1.9 GB"), ("3.8b", "~2.3 GB"),
            ("4b",   "~2.5 GB"), ("7b",   "~4.3 GB"), ("8b",   "~4.7 GB"),
            ("9b",   "~5.3 GB"), ("11b",  "~6.4 GB"), ("12b",  "~7.0 GB"),
            ("13b",  "~7.5 GB"), ("14b",  "~8.4 GB"), ("20b",  "~12 GB"),
            ("30b",  "~18 GB"),  ("32b",  "~19 GB"),  ("34b",  "~20 GB"),
            ("70b",  "~42 GB"),  ("72b",  "~43 GB"),
        ]:
            if pattern in nl:
                return size
        return "?"

    def _estimate_context(self, tags: list, name: str) -> str:
        nl = name.lower()
        for tag in tags:
            tl = tag.lower()
            if "128k" in tl or "131072" in tl:
                return "128K"
            if "32k" in tl or "32768" in tl:
                return "32K"
            if "8k" in tl or "8192" in tl:
                return "8K"
        if "llama-3" in nl or "qwen" in nl or "mistral" in nl:
            return "32K"
        if "phi-3.5" in nl or "phi3.5" in nl or "llama3.1" in nl or "llama3.2" in nl:
            return "128K"
        return "?"

    def invalidate_cache(self, source: Optional[str] = None):
        with self._lock:
            if source:
                keys = [k for k in self._cache if k.startswith(f"{source}:")]
                for k in keys:
                    del self._cache[k]
            else:
                self._cache.clear()
        logger.info(f"Katalog önbelleği temizlendi: {source or 'tümü'}")
