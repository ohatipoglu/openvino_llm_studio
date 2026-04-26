"""
modules/model_manager.py
OpenVINO model keşfi, tip tespiti ve yükleme yönetimi.
C:\\OpenVINO_LLM altındaki tüm modelleri otomatik tarar.
"""

import os
import json
import gc
import time
import logging
import threading
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

import psutil

logger = logging.getLogger(__name__)

from core.config import OPENVINO_MODELS_DIR, CACHE_DIR

OPENVINO_ROOT = Path(OPENVINO_MODELS_DIR)

# Model tipi tespiti için imzalar
MODEL_TYPE_SIGNATURES = {
    "vision": [
        "llava", "bakllava", "cogvlm", "internvl", "qwen-vl", "minicpm-v",
        "phi-3-vision", "paligemma", "idefics", "moondream", "nanollava",
        "vision", "visual", "vl", "multimodal", "image"
    ],
    "moe": [
        "mixtral", "moe", "mixtral", "deepseek-moe", "grok", "jamba",
        "olmoe", "qwen-moe", "phimoe", "mixture"
    ],
    "code": [
        "codellama", "codegemma", "deepseek-coder", "starcoder", "codestral",
        "code", "coder", "coding", "wizard-coder", "phind"
    ],
    "embedding": [
        "e5-", "bge-", "gte-", "minilm", "instructor", "embed", "sentence"
    ],
    "text": []  # default
}

ARCHITECTURE_MAP = {
    "llama": "LlamaForCausalLM",
    "mistral": "MistralForCausalLM",
    "phi": "PhiForCausalLM",
    "qwen": "Qwen2ForCausalLM",
    "gemma": "GemmaForCausalLM",
    "falcon": "FalconForCausalLM",
    "gpt2": "GPT2LMHeadModel",
    "bloom": "BloomForCausalLM",
    "mpt": "MPTForCausalLM",
    "internlm": "InternLMForCausalLM",
    "baichuan": "BaichuanForCausalLM",
    "chatglm": "ChatGLMForConditionalGeneration",
    "deepseek": "DeepseekForCausalLM",
    "yi": "YiForCausalLM",
    "orion": "OrionForCausalLM",
    "aquila": "AquilaForCausalLM",
}


@dataclass
class ModelInfo:
    name: str
    path: str
    model_type: str          # text / vision / moe / code / embedding
    architecture: str        # Transformer mimarisi
    has_tokenizer: bool
    has_config: bool
    config: dict = field(default_factory=dict)
    size_mb: float = 0.0
    openvino_files: list = field(default_factory=list)
    description: str = ""


class ModelScanner:
    """C:\\OpenVINO_LLM altındaki tüm modelleri tarar."""

    def __init__(self, root: Path = OPENVINO_ROOT):
        self.root = Path(root)

    def scan(self) -> list[ModelInfo]:
        if not self.root.exists():
            logger.warning(f"OpenVINO model dizini bulunamadı: {self.root}")
            return []

        models = []
        # Taramaya dahil edilmeyecek sistem/uygulama klasörleri
        ignore_dirs = {
            ".idea", ".cache", ".git", "gguf", "llama-server", 
            "ipex_ollama", "ipex_ollama_nightly", "openvino_llm_studio"
        }

        # Her alt dizini tara
        for subdir in self.root.iterdir():
            if not subdir.is_dir() or subdir.name in ignore_dirs or subdir.name.startswith("."):
                continue
                
            info = self._inspect_directory(subdir)
            if info:
                models.append(info)
                
            # Alt-alt dizinleri de tara (model içinde birden fazla variant)
            for subsubdir in subdir.iterdir():
                if subsubdir.is_dir() and not subsubdir.name.startswith("."):
                    info2 = self._inspect_directory(subsubdir, parent=subdir.name)
                    if info2:
                        models.append(info2)

        logger.info(f"{len(models)} OpenVINO modeli bulundu.")
        return models

    def _inspect_directory(self, path: Path, parent: str = "") -> Optional[ModelInfo]:
        """Bir dizinin OpenVINO modeli olup olmadığını ve tipini belirler."""
        files = list(path.iterdir()) if path.exists() else []
        file_names = [f.name.lower() for f in files]

        # OpenVINO model dosyası var mı?
        ov_files = [f.name for f in files if f.suffix in (".xml", ".bin")]
        if not ov_files:
            return None

        # Sadece 1-2 kb'lik rastgele xml dosyalarını (örneğin IDE kalıntıları) model sanmasını önlemek için 
        # en az bir tane .bin (ağırlık) dosyası veya openvino_model.xml arıyoruz.
        has_bin = any(f.endswith(".bin") for f in file_names)
        if not has_bin and "openvino_model.xml" not in file_names:
             return None

        # Config dosyasını oku
        config = {}
        config_path = path / "config.json"
        has_config = config_path.exists()
        if has_config:
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
            except Exception:
                pass

        # Tokenizer kontrolü
        has_tokenizer = any(
            f in file_names for f in
            ["tokenizer.json", "tokenizer_config.json", "vocab.json",
             "sentencepiece.bpe.model", "tokenizer.model"]
        )

        # Model tipi tespiti
        model_type = self._detect_type(path.name, config)

        # Mimari tespiti
        architecture = self._detect_architecture(path.name, config)

        # Boyut hesapla
        size_mb = sum(
            f.stat().st_size for f in files if f.is_file()
        ) / (1024 * 1024)

        name = f"{parent}/{path.name}" if parent else path.name

        return ModelInfo(
            name=name,
            path=str(path),
            model_type=model_type,
            architecture=architecture,
            has_tokenizer=has_tokenizer,
            has_config=has_config,
            config=config,
            size_mb=round(size_mb, 1),
            openvino_files=ov_files,
            description=self._build_description(model_type, architecture, config),
        )

    def _detect_type(self, name: str, config: dict) -> str:
        name_lower = name.lower()
        # Config'den architectures bilgisi
        arch = " ".join(config.get("architectures", [])).lower()
        combined = f"{name_lower} {arch}"

        for model_type, keywords in MODEL_TYPE_SIGNATURES.items():
            if model_type == "text":
                continue
            for kw in keywords:
                if kw in combined:
                    return model_type
        return "text"

    def _detect_architecture(self, name: str, config: dict) -> str:
        # Config'den
        archs = config.get("architectures", [])
        if archs:
            return archs[0]
        # İsimden tahmin
        name_lower = name.lower()
        for key, arch in ARCHITECTURE_MAP.items():
            if key in name_lower:
                return arch
        return "Unknown"

    def _build_description(self, model_type: str, architecture: str, config: dict) -> str:
        parts = [f"Tip: {model_type.upper()}"]
        if architecture and architecture != "Unknown":
            parts.append(f"Mimari: {architecture}")
        vocab = config.get("vocab_size")
        if vocab:
            parts.append(f"Vocab: {vocab:,}")
        ctx = config.get("max_position_embeddings") or config.get("n_positions")
        if ctx:
            parts.append(f"Context: {ctx:,}")
        return " | ".join(parts)


class ModelLoader:
    """OpenVINO-GenAI ile model yükler, bellek yönetimi yapar."""

    def __init__(self, db_manager=None):
        self.db = db_manager
        self._loaded_model = None
        self._loaded_path = None
        self._tokenizer = None
        self._lock = threading.Lock()

    def load(self, model_path: str, device: str = "CPU",
             session_id: str = "",
             ov_config: dict = None) -> tuple[bool, str]:
        """
        Model yükler. Önce mevcut modeli boşaltır.

        ov_config: OpenVINO plugin parametreleri (GPU bellek, cache, vb.)
        Örnek:
            {
              "GPU_MAX_ALLOC_PERCENT": "75",   # iGPU belleğinin max %'si
              "CACHE_DIR": "C:/OpenVINO_LLM/.cache",
              "NUM_STREAMS": "1",
              "PERFORMANCE_HINT": "LATENCY",
              "KV_CACHE_PRECISION": "u8",      # KV cache'i u8 ile tut (daha az RAM)
              "INFERENCE_PRECISION_HINT": "f16",
            }
        Returns: (success, message)
        """
        with self._lock:
            try:
                # Bellek kontrolü
                mem = psutil.virtual_memory()
                available_gb = mem.available / (1024 ** 3)
                if available_gb < 2.0:
                    msg = f"Yetersiz bellek: {available_gb:.1f} GB mevcut. En az 2 GB gerekli."
                    logger.warning(msg)
                    return False, msg

                # Mevcut modeli boşalt + GC
                self._unload()
                import gc
                gc.collect()

                # OpenVINO GenAI ile yükle
                try:
                    import openvino_genai as ov_genai
                    logger.info(f"Model yükleniyor: {model_path} [{device}] config={ov_config}")

                    # ov_config yoksa device'a göre akıllı default'lar uygula
                    cfg = self._build_ov_config(device, ov_config)

                    if cfg:
                        pipeline = ov_genai.LLMPipeline(model_path, device, **cfg)
                        logger.info(f"ov_config uygulandı: {cfg}")
                    else:
                        pipeline = ov_genai.LLMPipeline(model_path, device)

                    self._loaded_model = pipeline
                    self._loaded_path  = model_path
                    self._active_device = device
                    self._active_ov_config = cfg

                    # Token sayımı için tokenizer yükle (varsa)
                    try:
                        from transformers import AutoTokenizer
                        self._tokenizer = AutoTokenizer.from_pretrained(
                            model_path, local_files_only=True, trust_remote_code=False
                        )
                    except Exception:
                        self._tokenizer = None

                    mem_after = psutil.virtual_memory()
                    used_gb   = (mem.available - mem_after.available) / (1024 ** 3)
                    logger.info(f"Model yüklendi. Bellek kullanımı: ~{used_gb:.1f} GB")

                    if self.db and session_id:
                        self.db.log_general(
                            session_id, "INFO", "ModelLoader",
                            f"Model yüklendi: {Path(model_path).name}",
                            {"path": model_path, "device": device, "ov_config": cfg,
                             "ram_used_gb": round(used_gb, 2)}
                        )
                    return True, (
                        f"✅ {Path(model_path).name} yüklendi "
                        f"[{device}] | RAM: ~{used_gb:.1f} GB kullanıldı"
                    )

                except ImportError:
                    return self._load_via_optimum(model_path, device, session_id)

            except MemoryError as e:
                msg = "Bellek yetersiz — model yüklenemedi."
                logger.error(msg)
                if self.db:
                    self.db.log_error(session_id, "ModelLoader", e, {"path": model_path})
                return False, msg

            except Exception as e:
                msg = f"Model yükleme hatası: {type(e).__name__}: {e}"
                logger.error(msg, exc_info=True)
                if self.db:
                    self.db.log_error(session_id, "ModelLoader", e, {"path": model_path})
                return False, msg

    def _build_ov_config(self, device: str, user_config: dict = None) -> dict:
        """
        Device'a göre akıllı default ov_config oluştur.
        user_config ile override edilebilir.

        Intel Arc iGPU (paylaşımlı RAM) için kritik parametreler:
        - GPU_MAX_ALLOC_PERCENT: iGPU driver'ın ne kadar paylaşımlı RAM talep edeceği
          Default 100% → driver tüm paylaşımlı RAM'i ayırmaya çalışır → dxgmms2.sys crash
          75% → 32 GB paylaşımlı RAM'den max ~24 GB kullanır, sistem kararlı kalır
        - CACHE_DIR: compiled model kernel cache (yeniden yüklemede hız)
        - NUM_STREAMS: büyük modellerde 1 olmalı (bellek tasarrufu)
        - PERFORMANCE_HINT: LATENCY = tek kullanıcı için optimal, düşük bellek
        - KV_CACHE_PRECISION: u8 = KV cache'i 8-bit integer ile tut (~2x küçük)
        """
        import os

        dev_upper = device.upper()
        cfg = {}

        if dev_upper in ("GPU", "AUTO"):
            cfg = {
                # iGPU paylaşımlı RAM'in max %75'ini kullan
                # 32 GB × 0.75 = 24 GB max → dxgmms2.sys crash önlenir
                "GPU_MAX_ALLOC_PERCENT": "75",
                # Compiled kernel cache → yeniden yüklemede hızlı
                "CACHE_DIR": str(CACHE_DIR),
                # Tek stream → büyük modelde bellek tasarrufu
                "NUM_STREAMS": "1",
                # Latency modu → single-batch, düşük overhead
                "PERFORMANCE_HINT": "LATENCY",
                # KV cache'i u8 ile tut → ~2x daha az VRAM
                "KV_CACHE_PRECISION": "u8",
                # f16 inference → int4 model için yeterli, bellek tasarrufu
                "INFERENCE_PRECISION_HINT": "f16",
            }
        elif dev_upper == "CPU":
            # Intel vPRO için CPU optimizasyonları
            cpu_count = os.cpu_count() or 8
            threads   = max(4, cpu_count - 2)  # 2 çekirdeği sistem için bırak
            cfg = {
                "INFERENCE_NUM_THREADS": str(threads),
                "CPU_BIND_THREAD": "YES",
                # Compiled model cache
                "CACHE_DIR": str(CACHE_DIR),
                "PERFORMANCE_HINT": "LATENCY",
                "NUM_STREAMS": "1",
            }

        # Kullanıcı override'ları uygula
        if user_config:
            cfg.update(user_config)

        return cfg

    def _load_via_optimum(self, model_path: str, device: str, session_id: str):
        """optimum-intel ile yedek yükleme."""
        try:
            from optimum.intel import OVModelForCausalLM
            from transformers import AutoTokenizer

            logger.info("optimum-intel ile yükleniyor...")
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = OVModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                device=device.lower()
            )
            self._loaded_model = (model, tokenizer)
            self._loaded_path = model_path
            self._tokenizer = tokenizer
            return True, f"Model yüklendi (optimum): {Path(model_path).name}"
        except Exception as e:
            msg = f"optimum-intel yükleme hatası: {str(e)}"
            logger.error(msg)
            if self.db:
                self.db.log_error(session_id, "ModelLoader", e)
            return False, msg

    def _unload(self):
        """Mevcut modeli bellekten çıkar."""
        if self._loaded_model is not None:
            try:
                del self._loaded_model
                self._loaded_model = None
                self._loaded_path = None
                self._tokenizer = None
                gc.collect()
                logger.info("Önceki model bellekten temizlendi.")
            except Exception as e:
                logger.warning(f"Model boşaltma uyarısı: {e}")

    def generate(self, prompt: str, params: dict, session_id: str = "",
                 raw_prompt: str = "", system_prompt: str = "") -> tuple[str, dict]:
        """
        Metin üretir. Instruct modeller için chat template uygular.
        Returns: (response_text, metrics_dict)
        """
        if self._loaded_model is None:
            return "", {"error": "Model yüklü değil."}

        with self._lock:
            try:
                start = time.time()
                model_name_lower = Path(self._loaded_path).name.lower() if self._loaded_path else ""

                # Instruct modeller için chat template formatı
                formatted_prompt = self._apply_chat_template(
                    prompt, system_prompt, model_name_lower
                )

                # openvino_genai pipeline
                if hasattr(self._loaded_model, 'generate'):
                    try:
                        import openvino_genai as ov_genai
                        config = ov_genai.GenerationConfig()
                        config.max_new_tokens = params.get("max_tokens", 512)
                        config.temperature = params.get("temperature", 0.7)
                        config.top_p = params.get("top_p", 0.9)
                        config.top_k = params.get("top_k", 50)
                        config.repetition_penalty = params.get("repetition_penalty", 1.1)
                        config.do_sample = params.get("temperature", 0.7) > 0

                        response = self._loaded_model.generate(formatted_prompt, config)
                    except Exception as e:
                        logger.warning(f"GenerationConfig başarısız, varsayılanla deneniyor: {e}")
                        response = self._loaded_model.generate(formatted_prompt)

                # optimum-intel (model, tokenizer) tuple
                elif isinstance(self._loaded_model, tuple):
                    model, tokenizer = self._loaded_model
                    inputs = tokenizer(formatted_prompt, return_tensors="pt")
                    output = model.generate(
                        **inputs,
                        max_new_tokens=params.get("max_tokens", 512),
                        temperature=params.get("temperature", 0.7),
                        top_p=params.get("top_p", 0.9),
                        top_k=params.get("top_k", 50),
                        repetition_penalty=params.get("repetition_penalty", 1.1),
                        do_sample=params.get("temperature", 0.7) > 0,
                    )
                    response = tokenizer.decode(
                        output[0][inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True
                    )
                else:
                    return "", {"error": "Bilinmeyen model formatı."}

                # Prompt tekrarını yanıttan temizle
                response_text = self._clean_response(str(response), formatted_prompt, prompt)

                elapsed_ms = (time.time() - start) * 1000
                input_tokens = self._count_tokens(formatted_prompt)
                output_tokens = self._count_tokens(response_text)
                tps = output_tokens / max(elapsed_ms / 1000, 0.001)

                metrics = {
                    "duration_ms": round(elapsed_ms, 1),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "tokens_per_second": round(tps, 2),
                }

                if self.db:
                    model_name = Path(self._loaded_path).name if self._loaded_path else "unknown"
                    self.db.log_llm(
                        session_id=session_id,
                        model_name=model_name,
                        model_type="text",
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
                logger.error(f"Generate hatası: {e}")
                if self.db:
                    self.db.log_error(session_id, "ModelLoader.generate", e)
                return f"[HATA] {str(e)}", {"error": str(e)}

    def _count_tokens(self, text: str) -> int:
        """Token sayısını ölç. Tokenizer yoksa karakter tabanlı tahmin kullan."""
        if self._tokenizer is not None:
            try:
                return len(self._tokenizer.encode(text, add_special_tokens=False))
            except Exception:
                pass
        # Fallback: ortalama ~4 karakter/token (GPT-style tokenizer tahmini)
        return max(1, len(text) // 4)

    def _apply_chat_template(self, prompt: str, system_prompt: str, model_name: str) -> str:
        """
        Model tipine göre doğru chat template uygular.
        Instruct modeller için önemli — aksi halde model prompt'u tekrar eder.
        """
        sys = system_prompt.strip() if system_prompt else "You are a helpful assistant."

        # Falcon Instruct
        if "falcon" in model_name:
            if system_prompt:
                return f"System: {sys}\nUser: {prompt}\nAssistant:"
            return f"User: {prompt}\nAssistant:"

        # Open LLaMA / base LLaMA — instruct değil, düz prompt
        if "open_llama" in model_name or "open-llama" in model_name:
            return prompt  # Base model, template uygulanmaz

        # Llama-2 Instruct
        if "llama-2" in model_name or "llama2" in model_name:
            if "instruct" in model_name or "chat" in model_name:
                return f"[INST] <<SYS>>\n{sys}\n<</SYS>>\n\n{prompt} [/INST]"
            return prompt  # base model
        if "llama-3" in model_name or "llama3" in model_name or "meta-llama" in model_name:
            return (f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{sys}"
                    f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n{prompt}"
                    f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n")

        # Mistral / Mixtral Instruct
        if "mistral" in model_name or "mixtral" in model_name:
            return f"<s>[INST] {prompt} [/INST]"

        # Phi-3 Instruct
        if "phi-3" in model_name or "phi3" in model_name:
            return f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"

        # Phi-2
        if "phi-2" in model_name or "phi2" in model_name:
            return f"Instruct: {prompt}\nOutput:"

        # Qwen / Qwen2
        if "qwen" in model_name:
            return (f"<|im_start|>system\n{sys}<|im_end|>\n"
                    f"<|im_start|>user\n{prompt}<|im_end|>\n"
                    f"<|im_start|>assistant\n")

        # Gemma Instruct
        if "gemma" in model_name:
            return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"

        # ChatGLM
        if "chatglm" in model_name or "glm" in model_name:
            return f"[Round 1]\n\n问：{prompt}\n\n答："

        # DeepSeek
        if "deepseek" in model_name:
            return f"<|begin_of_sentence|>User: {prompt}\n\nAssistant:"

        # Vicuna / FastChat style
        if "vicuna" in model_name or "openchat" in model_name:
            return f"USER: {prompt}\nASSISTANT:"

        # Genel instruct modeller
        if any(k in model_name for k in ["instruct", "chat", "it", "-it-"]):
            if system_prompt:
                return f"<s>[INST] {sys}\n\n{prompt} [/INST]"
            return f"<s>[INST] {prompt} [/INST]"

        # Base model — düz prompt
        if system_prompt:
            return f"{sys}\n\n{prompt}"
        return prompt

    def _clean_response(self, response: str, formatted_prompt: str, raw_prompt: str) -> str:
        """
        Model çıktısından prompt tekrarını ve özel tokenleri temizler.
        """
        text = response.strip()

        # Eğer yanıt formatted_prompt ile başlıyorsa çıkar
        if text.startswith(formatted_prompt):
            text = text[len(formatted_prompt):].strip()

        # Eğer ham prompt ile başlıyorsa çıkar
        if text.lower().startswith(raw_prompt.lower()[:50]):
            text = text[len(raw_prompt):].strip()

        # Yaygın özel tokenleri temizle
        for token in ["</s>", "<|endoftext|>", "<|im_end|>", "<|eot_id|>",
                       "<end_of_turn>", "[/INST]", "<|end|>"]:
            text = text.replace(token, "").strip()

        # Sonsuz tekrar tespiti: aynı cümle 3+ kez geçiyorsa kes
        lines = text.split("\n")
        seen = {}
        clean_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                clean_lines.append(line)
                continue
            seen[stripped] = seen.get(stripped, 0) + 1
            if seen[stripped] <= 2:  # En fazla 2 kez izin ver
                clean_lines.append(line)
            else:
                break  # Tekrar başladı, kes

        return "\n".join(clean_lines).strip()



    @property
    def is_loaded(self) -> bool:
        return self._loaded_model is not None

    @property
    def loaded_model_name(self) -> str:
        return Path(self._loaded_path).name if self._loaded_path else ""

    def get_memory_info(self) -> dict:
        mem = psutil.virtual_memory()
        return {
            "total_gb": round(mem.total / 1024**3, 1),
            "used_gb": round(mem.used / 1024**3, 1),
            "available_gb": round(mem.available / 1024**3, 1),
            "percent": mem.percent,
        }
