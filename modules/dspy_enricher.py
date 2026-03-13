"""
modules/dspy_enricher.py
DSPy ile prompt zenginleştirme ve LLM tabanlı otomatik mod seçimi.
Mod seçimi sabit keyword listesiyle değil, LLM'e sorarak yapılır.
"""

import time
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


DSPY_MODES = {
    "ChainOfThought": "Karmaşık, analitik veya açıklama gerektiren sorular. Adım adım düşünme gerektirir.",
    "ReAct":          "Güncel bilgi, haber veya araştırma gerektiren sorular.",
    "ProgramOfThought": "Matematik, hesaplama veya kod gerektiren sorular.",
    "MultiChainComparison": "Karşılaştırma, eleştiri, avantaj/dezavantaj veya farklı bakış açısı isteyen sorular.",
    "Summarize":      "Bir konuyu özetleme veya kısa açıklama isteyen sorular.",
    "Predict":        "Basit, kısa ve doğrudan yanıt gerektiren olgusal sorular.",
}


@dataclass
class EnrichmentResult:
    original_prompt: str
    enriched_prompt: str
    mode: str
    mode_reason: str
    steps: list
    duration_ms: float


class DSPyEnricher:
    """
    DSPy ile prompt zenginleştirme.
    Mod seçimi için yüklü LLM'i kullanır (dspy.Predict ile).
    DSPy veya LLM yoksa basit fallback uygular.
    """

    def __init__(self, db_manager=None):
        self.db = db_manager
        self._dspy_available = False
        self._lm = None
        self._mode_predictor = None
        self._try_init_dspy()

    def _try_init_dspy(self):
        try:
            import dspy
            self._dspy = dspy
            self._dspy_available = True
            logger.info("DSPy yüklendi. LLM bağlandığında mod seçimi LLM ile yapılacak.")
        except ImportError:
            logger.warning("DSPy bulunamadı. Fallback mod aktif.")

    def configure_lm(self, model_path: str, device: str = "CPU"):
        """
        Model yüklenince çağrılır. DSPy LM bağlantısı yerine
        direkt LLM sınıflandırması için loader referansını saklar.
        DSPy'ın JSONAdapter'ı küçük modellerde güvenilmez olduğu için
        doğrudan generate() kullanıyoruz.
        """
        loader = getattr(self, "_loader", None)
        if loader is None or not loader.is_loaded:
            logger.warning("DSPy LM: loader hazır değil.")
            return False
        logger.info(f"DSPy direkt LLM modu aktif: {model_path}")
        self._mode_predictor = "direct"   # Flag: direkt LLM kullan
        return True

    def set_loader(self, loader):
        """ModelLoader referansını set et (orchestrator tarafından çağrılır)."""
        self._loader = loader

    def enrich(self, prompt: str, search_context: str = "",
               session_id: str = "") -> EnrichmentResult:
        start = time.time()
        steps = []

        # Mod seçimi: önce DSPy+LLM, yoksa fallback
        mode, reason = self._select_mode_via_llm(prompt, steps)

        # Prompt zenginleştirme
        enriched, steps = self._apply_template(prompt, mode, search_context, steps)

        duration_ms = (time.time() - start) * 1000

        result = EnrichmentResult(
            original_prompt=prompt,
            enriched_prompt=enriched,
            mode=mode,
            mode_reason=reason,
            steps=steps,
            duration_ms=duration_ms,
        )

        if self.db:
            self.db.log_dspy(
                session_id=session_id,
                original_prompt=prompt,
                detected_mode=mode,
                mode_reason=reason,
                enriched_prompt=enriched,
                dspy_steps=steps,
                duration_ms=duration_ms,
            )

        return result

    # DSPy classify için maksimum süre (saniye). Bu süreyi aşan büyük modellerde
    # rule_fallback kullanılır — asıl generate() için zaman harcamayız.
    _CLASSIFY_TIMEOUT_S = 15

    def _select_mode_via_llm(self, prompt: str, steps: list) -> tuple[str, str]:
        """LLM ile mod seç. Başarısız olursa fallback."""

        # DSPy Predict yerine direkt LLM çağrısı (JSONAdapter sorununu atlar)
        if self._mode_predictor == "direct":
            loader = getattr(self, "_loader", None)
            if loader and loader.is_loaded:
                # Büyük modellerde sınıflandırma çok yavaş olabilir;
                # timeout'u aşarsa rule_fallback kullanırız.
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    future = ex.submit(self._fallback_llm_classify, prompt, steps)
                    try:
                        return future.result(timeout=self._CLASSIFY_TIMEOUT_S)
                    except concurrent.futures.TimeoutError:
                        logger.warning(
                            f"LLM classify {self._CLASSIFY_TIMEOUT_S}s'de tamamlanamadı, "
                            "rule_fallback kullanılıyor."
                        )
                        steps.append({"action": "classify_timeout"})
                        future.cancel()
                        return self._rule_fallback(prompt, steps)

        # DSPy predictor hazırsa dene (gelecekte daha yetenekli modeller için)
        if self._dspy_available and self._mode_predictor and self._mode_predictor != "direct":
            try:
                steps.append({"action": "dspy_llm_mode_selection", "prompt": prompt[:100]})
                result = self._mode_predictor(question=prompt)
                raw_mode = str(result.mode).strip()
                reason = str(result.reason).strip()
                matched = self._fuzzy_match_mode(raw_mode)
                steps.append({"action": "llm_selected_mode", "raw": raw_mode, "matched": matched})
                logger.info(f"DSPy Predict mod: '{raw_mode}' → '{matched}'")
                return matched, reason
            except Exception as e:
                logger.warning(f"DSPy Predict başarısız, fallback: {e}")
                steps.append({"action": "dspy_llm_failed", "error": str(e)})

        # Son çare
        return self._rule_fallback(prompt, steps)

    def _fallback_llm_classify(self, prompt: str, steps: list) -> tuple[str, str]:
        """
        DSPy olmadan direkt LLM'e kısa sınıflandırma sorusu gönder.
        Sadece kullanıcının ham promptu kullanılır — arama sonuçları dahil edilmez.

        NOT: OpenVINO büyük modeller (20B+) bu sınıflandırmayı yanlış yapabiliyor.
        Bu durumda rule_fallback'e geçilir ve hiç zaman harcanmaz.
        """
        try:
            loader = self._loader
            if loader is None or not loader.is_loaded:
                return self._rule_fallback(prompt, steps)

            # generate_raw varsa kullan (daha hızlı, chat template uygulamaz)
            gen_fn = getattr(loader, "generate_raw", None) or loader.generate

            mode_names = ", ".join(DSPY_MODES.keys())
            # Kısa, net, tek-kelime cevap bekleyen prompt
            classify_prompt = (
                f"Classify this question. Reply with ONLY one word from this list: {mode_names}\n"
                f"Question: {prompt}\n"
                f"Answer:"
            )

            if gen_fn is loader.generate:
                response, _ = gen_fn(
                    prompt=classify_prompt,
                    params={"max_tokens": 8, "temperature": 0.0},
                )
            else:
                response = gen_fn(
                    prompt=classify_prompt,
                    params={"max_tokens": 8, "temperature": 0.0},
                )

            raw_resp = str(response).strip()
            # İlk kelimeyi al; eğer yanıt çok uzunsa (model classify etmedi) rule_fallback
            words = raw_resp.split()
            if not words or len(raw_resp) > 80:
                logger.warning(
                    f"LLM classify yanıtı uygunsuz ('{raw_resp[:60]}'), rule_fallback kullanılıyor."
                )
                return self._rule_fallback(prompt, steps)

            raw = words[0].strip(".,:\n\"'")
            matched = self._fuzzy_match_mode(raw)
            reason = f"LLM sınıflandırması: '{raw}'"
            steps.append({"action": "fallback_llm_classify", "raw": raw, "matched": matched})
            logger.info(f"Fallback LLM mod: '{raw}' → '{matched}'")
            return matched, reason

        except Exception as e:
            logger.warning(f"Fallback LLM classify hatası: {e}")
            return self._rule_fallback(prompt, steps)

    def _fuzzy_match_mode(self, raw: str) -> str:
        """LLM çıktısından geçerli mod adı çıkar."""
        raw_lower = raw.lower()
        for mode in DSPY_MODES:
            if mode.lower() in raw_lower or raw_lower in mode.lower():
                return mode
        # Kısmi eşleşme
        for mode in DSPY_MODES:
            for word in mode.lower().split():
                if word in raw_lower:
                    return mode
        return "ChainOfThought"  # tanınmayan çıktı için güvenli default

    def _rule_fallback(self, prompt: str, steps: list) -> tuple[str, str]:
        """
        Ne DSPy ne LLM varsa son çare: minimum kural seti.
        Sadece çok belirgin kategoriler için, yoksa ChainOfThought.
        """
        p = prompt.lower()
        steps.append({"action": "rule_fallback"})

        if any(k in p for k in ["özetle", "summarize", "tldr", "kısaca"]):
            return "Summarize", "Özetleme kalıbı tespit edildi."
        if any(k in p for k in ["hesapla", "calculate", "kod yaz", "write code"]):
            return "ProgramOfThought", "Hesaplama/kod kalıbı tespit edildi."
        if any(k in p for k in ["güncel", "haber", "latest", "news"]):
            return "ReAct", "Güncel bilgi kalıbı tespit edildi."
        # Soru işareti veya karmaşık yapı → CoT
        if "?" in prompt or len(prompt.split()) > 8:
            return "ChainOfThought", "Karmaşık soru yapısı, adım adım düşünme uygulandı."
        return "Predict", "Kısa ve basit ifade."

    def _apply_template(self, prompt: str, mode: str,
                        search_context: str, steps: list) -> tuple[str, list]:
        """
        Seçilen moda göre prompt'u zenginleştir.
        Bağlam sona yerleştirilir — base modellerde başa koyunca
        model bağlamı devam ettirmeye çalışır, bu yanlış çıktı üretir.
        """
        ctx = f"\n\n{search_context}" if search_context else ""

        templates = {
            "ChainOfThought": (
                f"Aşağıdaki soruyu adım adım düşünerek Türkçe yanıtla:\n\n"
                f"Soru: {prompt}{ctx}\n\nYanıt:"
            ),
            "ReAct": (
                f"Aşağıdaki soruyu arama sonuçlarına dayanarak Türkçe yanıtla:\n\n"
                f"Soru: {prompt}{ctx}\n\nYanıt:"
            ),
            "ProgramOfThought": (
                f"Aşağıdaki problemi çöz, gerekirse formül veya kod kullan:\n\n"
                f"Problem: {prompt}{ctx}\n\nÇözüm:"
            ),
            "MultiChainComparison": (
                f"Aşağıdaki konuyu farklı açılardan Türkçe değerlendir:\n\n"
                f"Konu: {prompt}{ctx}\n\nDeğerlendirme:"
            ),
            "Summarize": (
                f"Aşağıdaki konuyu kısaca Türkçe özetle:\n\n"
                f"Konu: {prompt}{ctx}\n\nÖzet:"
            ),
            "Predict": (
                f"{prompt}{ctx}"
            ),
        }

        enriched = templates.get(mode, templates["ChainOfThought"])
        steps.append({"action": "template_applied", "mode": mode})
        return enriched, steps

