"""
ui/app_modern.py

Modernized OpenVINO LLM Studio UI with:
- Progressive disclosure
- User mode selector (Beginner/Advanced/Expert)
- Workflow-based navigation (Steps)
- Preset profiles
- Real-time status dashboard
- Contextual help tooltips
- Clean, focused layout
"""

import os
import socket
import sys
import json
import logging
import threading
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import gradio as gr
from core.orchestrator import Orchestrator
from core.config import OPENVINO_LLM_HOME, CACHE_DIR, GGUF_DIR, LOG_DIR

# Ensure logs directory exists
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "studio_modern.log", encoding="utf-8"),
    ]
)
logger = logging.getLogger(__name__)

# Global orchestrator
_orch_lock = threading.Lock()
_orch = None

def _get_orch() -> Orchestrator:
    global _orch
    with _orch_lock:
        if _orch is None:
            _orch = Orchestrator()
        return _orch


# ═══════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def get_backend_status_display():
    """Get formatted backend status."""
    orch = _get_orch()
    status = orch.get_backend_status()
    
    lines = ["### 🔌 Backend Durumu\n"]
    for backend, stat in status.items():
        icon = "✅" if "✅" in stat else "❌"
        lines.append(f"- **{backend.title()}**: {icon}")
    
    return "\n".join(lines)


def get_system_status():
    """Get system status for dashboard."""
    import psutil
    
    cpu = psutil.cpu_percent(interval=0.1)
    mem = psutil.virtual_memory()
    
    orch = _get_orch()
    model_loaded = orch.is_model_loaded
    model_name = orch._active_loader.loaded_model_name if model_loaded else "Yok"
    
    return {
        "cpu": cpu,
        "memory": mem.percent,
        "model_loaded": model_loaded,
        "model_name": model_name,
        "backend": orch._active_backend,
    }


def format_status_html(status: dict) -> str:
    """Format status as HTML badge."""
    model_status = (
        f'<span style="color:#22c55e">✅ {status["model_name"]}</span>'
        if status["model_loaded"] else
        '<span style="color:#ef4444">❌ Model Yok</span>'
    )
    
    cpu_color = "#22c55e" if status["cpu"] < 50 else "#f59e0b" if status["cpu"] < 80 else "#ef4444"
    mem_color = "#22c55e" if status["memory"] < 50 else "#f59e0b" if status["memory"] < 80 else "#ef4444"
    
    return f"""
    <div style="display: flex; gap: 16px; flex-wrap: wrap; font-family: monospace; font-size: 0.9em;">
        <div style="padding: 8px 12px; background: #f0fdf4; border-radius: 6px; border-left: 4px solid #22c55e;">
            <b>🤖 Model:</b> {model_status}
        </div>
        <div style="padding: 8px 12px; background: #eff6ff; border-radius: 6px; border-left: 4px solid #3b82f6;">
            <b>⚡ Backend:</b> {status["backend"].upper()}
        </div>
        <div style="padding: 8px 12px; background: #fef3c7; border-radius: 6px; border-left: 4px solid #f59e0b;">
            <b>💻 CPU:</b> <span style="color:{cpu_color}">{status["cpu"]:.1f}%</span>
        </div>
        <div style="padding: 8px 12px; background: #fce7f3; border-radius: 6px; border-left: 4px solid #ec4899;">
            <b>💾 RAM:</b> <span style="color:{mem_color}">{status["memory"]:.1f}%</span>
        </div>
    </div>
    """


def apply_preset(preset_name):
    """Apply preset configuration."""
    presets = {
        "🚀 Hızlı (Düşük Kalite)": {
            "temperature": 0.5,
            "max_tokens": 256,
            "top_p": 0.8,
            "top_k": 40,
            "repetition_penalty": 1.05,
        },
        "⚖️ Dengeli (Önerilen)": {
            "temperature": 0.7,
            "max_tokens": 512,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1,
        },
        "🎨 Kaliteli (Yüksek Kalite)": {
            "temperature": 0.9,
            "max_tokens": 1024,
            "top_p": 0.95,
            "top_k": 100,
            "repetition_penalty": 1.15,
        },
    }
    
    if preset_name in presets:
        p = presets[preset_name]
        return [
            gr.update(value=p["temperature"]),
            gr.update(value=p["max_tokens"]),
            gr.update(value=p["top_p"]),
            gr.update(value=p["top_k"]),
            gr.update(value=p["repetition_penalty"]),
        ]
    
    # Custom
    return [gr.update() for _ in range(5)]


def refresh_models():
    """Refresh model list."""
    orch = _get_orch()
    choices = orch.get_model_choices()
    if choices:
        return gr.update(choices=choices, value=choices[0]), f"✅ {len(choices)} model bulundu."
    return gr.update(choices=[], value=None), "⚠️ Model bulunamadı."


def load_model_action(model_choice, device, gpu_max_alloc, kv_cache_prec, num_streams, perf_hint, cache_dir):
    """Load model action."""
    orch = _get_orch()
    if not model_choice:
        return "❌ Model seçilmedi.", gr.update(interactive=False)
    
    ov_config = None
    if device.upper() in ("GPU", "AUTO"):
        ov_config = {
            "GPU_MAX_ALLOC_PERCENT": str(int(gpu_max_alloc)),
            "KV_CACHE_PRECISION": kv_cache_prec,
            "NUM_STREAMS": str(int(num_streams)),
            "PERFORMANCE_HINT": perf_hint,
        }
        if cache_dir.strip():
            ov_config["CACHE_DIR"] = cache_dir.strip()
    elif device.upper() == "CPU":
        cpu_count = os.cpu_count() or 8
        ov_config = {
            "INFERENCE_NUM_THREADS": str(max(4, cpu_count - 2)),
            "NUM_STREAMS": str(int(num_streams)),
            "PERFORMANCE_HINT": perf_hint,
            "CPU_BIND_THREAD": "YES",
        }
    
    success, msg = orch.load_model(model_choice, device, ov_config=ov_config)
    return msg, gr.update(interactive=success)


def run_inference(prompt, system_prompt, enable_search, enable_dspy,
                  num_search_results, search_region, temperature, max_tokens,
                  top_p, top_k, repetition_penalty, history):
    """Run inference pipeline."""
    orch = _get_orch()
    
    if not prompt.strip():
        yield history, "⚠️ Prompt boş olamaz.", gr.update()
        return
    
    if not orch.is_model_loaded:
        yield history, "❌ Model yüklü değil. Lütfen önce model seçin ve yükleyin.", gr.update()
        return
    
    params = {
        "temperature": temperature,
        "max_tokens": int(max_tokens),
        "top_p": top_p,
        "top_k": int(top_k),
        "repetition_penalty": repetition_penalty,
    }
    
    accumulated = ""
    
    for chunk in orch.run_pipeline(
        prompt=prompt,
        params=params,
        enable_search=enable_search,
        enable_dspy=enable_dspy,
        num_search_results=int(num_search_results),
        system_prompt=system_prompt,
    ):
        accumulated += chunk
        yield history, accumulated, gr.update(value=format_status_html(get_system_status()))
    
    history = list(history)
    history.append({"role": "user", "content": prompt})
    history.append({"role": "assistant", "content": accumulated})
    yield history, accumulated, gr.update(value=format_status_html(get_system_status()))


def update_ui_for_mode(mode):
    """Update UI visibility based on user mode."""
    # Returns visibility settings for advanced components
    if mode == "🔰 Basit":
        return [
            gr.update(visible=False),  # GPU ayarları
            gr.update(visible=False),  # LLM parametreleri detaylı
            gr.update(visible=False),  # Sistem prompt
        ]
    elif mode == "⚙️ Orta":
        return [
            gr.update(visible=False),  # GPU ayarları hala gizli
            gr.update(visible=True),   # LLM parametreleri
            gr.update(visible=True),   # Sistem prompt
        ]
    else:  # Expert
        return [
            gr.update(visible=True),   # GPU ayarları
            gr.update(visible=True),   # LLM parametreleri
            gr.update(visible=True),   # Sistem prompt
        ]


# ═══════════════════════════════════════════════════════════════════
# CSS & THEME
# ═══════════════════════════════════════════════════════════════════

CSS = """
#header {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #1e3a5f 100%);
    padding: 24px;
    border-radius: 12px;
    margin-bottom: 20px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
#header h1 {
    color: #e0e7ff;
    font-size: 2em;
    margin: 0;
    font-weight: 700;
}
#header p {
    color: #a5b4fc;
    margin: 8px 0 0 0;
    font-size: 1em;
}
.status-bar {
    background: linear-gradient(to right, #f0fdf4, #eff6ff);
    padding: 12px 16px;
    border-radius: 8px;
    margin: 16px 0;
    border: 1px solid #e5e7eb;
}
.model-status {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85em;
}
#response-box textarea {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9em;
}
.step-container {
    background: #f9fafb;
    padding: 16px;
    border-radius: 8px;
    margin: 8px 0;
}
"""

THEME = gr.themes.Base(primary_hue="indigo", neutral_hue="slate")


# ═══════════════════════════════════════════════════════════════════
# MAIN UI BUILDER
# ═══════════════════════════════════════════════════════════════════

def build_modern_ui():
    """Build modernized UI with all improvements."""
    
    with gr.Blocks(title="OpenVINO LLM Studio - Modern") as demo:
        
        # Header
        gr.HTML("""
        <div id="header">
            <h1>⚡ OpenVINO LLM Studio</h1>
            <p>Modern UI · Intel OpenVINO · Ollama · GGUF/SYCL · DSPy · Web Search</p>
        </div>
        """)
        
        # Status Bar (always visible)
        status_html = gr.HTML(
            value=format_status_html(get_system_status()),
            elem_classes=["status-bar"]
        )
        
        # Mode Selector
        with gr.Row():
            mode_selector = gr.Radio(
                label="🎯 Kullanıcı Modu",
                choices=["🔰 Basit", "⚙️ Orta", "🔬 Expert"],
                value="⚙️ Orta",
                info="Basit: Sadece temel ayarlar | Orta: Önerilen | Expert: Tüm ayarlar",
                interactive=True
            )
        
        # Main Steps-based Navigation
        with gr.Tabs() as workflow_tabs:
            
            # ═══════════════════════════════════════
            # STEP 1: MODEL SELECTION
            # ═══════════════════════════════════════
            with gr.TabItem("1️⃣ Model Seçimi", id="step1"):
                gr.Markdown("### 🤖 Model Seçimi ve Yükleme")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        # Backend Selection
                        backend_radio = gr.Radio(
                            label="Backend",
                            choices=["openvino", "ollama", "ipex"],
                            value="openvino",
                            info="OpenVINO: .xml/.bin | Ollama: yerel sunucu | llama-cpp: GGUF"
                        )
                        
                        backend_status_md = gr.Markdown(value=get_backend_status_display())
                        refresh_backend_btn = gr.Button("🔄 Backend Durumunu Yenile", size="sm")
                        
                        gr.Markdown("---")
                        
                        # Model Selection
                        with gr.Row():
                            model_dropdown = gr.Dropdown(
                                label="Model",
                                choices=[],
                                interactive=True,
                                allow_custom_value=True
                            )
                            refresh_models_btn = gr.Button("🔄", scale=0, size="sm")
                        
                        device_radio = gr.Radio(
                            label="Cihaz",
                            choices=["CPU", "GPU", "AUTO"],
                            value="CPU",
                            info="GPU: Intel Arc iGPU | CPU: Tüm işlemciler"
                        )
                        
                        load_model_btn = gr.Button("⚡ Modeli Yükle", variant="primary", size="lg")
                        load_status = gr.Textbox(
                            label="Yükleme Durumu",
                            value="Model yüklenmedi.",
                            interactive=False,
                            elem_classes=["model-status"]
                        )
                    
                    with gr.Column(scale=1):
                        # Advanced GPU Settings (hidden in simple mode)
                        with gr.Accordion("⚙️ Gelişmiş GPU/CPU Ayarları", open=False, visible=True) as gpu_accordion:
                            gr.Markdown(
                                "**Intel Arc iGPU** için optimize ayarlar.\n\n"
                                "💡 **İpucu:** `dxgmms2.sys` crash önleme için GPU Max Alloc %75'ten başlayın."
                            )
                            
                            gpu_max_alloc = gr.Slider(
                                label="GPU Max Alloc %",
                                minimum=40, maximum=95, value=75, step=5,
                                info="ⓘ Paylaşımlı RAM limiti. Düşük = crash riski az, yüksek = daha fazla VRAM"
                            )
                            
                            kv_cache_prec = gr.Radio(
                                label="KV Cache Precision",
                                choices=["f16", "u8"],
                                value="u8",
                                info="ⓘ u8 = ~2x daha az VRAM. Büyük modellerde önerilen."
                            )
                            
                            num_streams = gr.Slider(
                                label="Num Streams",
                                minimum=1, maximum=4, value=1, step=1,
                                info="ⓘ Büyük modellerde 1 tut. Yüksek değer bellek artırır."
                            )
                            
                            perf_hint = gr.Radio(
                                label="Performance Hint",
                                choices=["LATENCY", "THROUGHPUT"],
                                value="LATENCY",
                                info="ⓘ LATENCY = tek kullanıcı, düşük overhead. THROUGHPUT = batch."
                            )
                            
                            cache_dir = gr.Textbox(
                                label="Cache Dizini",
                                value=str(CACHE_DIR),
                                info="ⓘ Compiled kernel cache dizini"
                            )
                
                # Ollama Quick Download
                with gr.Accordion("🦙 Ollama Model İndir", open=False):
                    with gr.Row():
                        ollama_pull_input = gr.Textbox(
                            label="Model ID",
                            placeholder="qwen2.5:7b",
                            scale=3
                        )
                        ollama_pull_btn = gr.Button("⬇️ İndir", variant="secondary", scale=1)
                    ollama_pull_status = gr.Textbox(
                        label="Durum",
                        interactive=False,
                        max_lines=2
                    )
            
            # ═══════════════════════════════════════
            # STEP 2: CONFIGURATION
            # ═══════════════════════════════════════
            with gr.TabItem("2️⃣ Ayarlar", id="step2"):
                gr.Markdown("### ⚙️ Konfigürasyon")
                
                # Preset Selector
                preset_selector = gr.Radio(
                    label="🎯 Hızlı Ayar (Preset)",
                    choices=["🚀 Hızlı (Düşük Kalite)", "⚖️ Dengeli (Önerilen)", "🎨 Kaliteli (Yüksek Kalite)"],
                    value="⚖️ Dengeli (Önerilen)",
                    info="Hazır ayarlardan seçin veya manuel ayarlayın"
                )
                
                gr.Markdown("---")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        # Search & DSPy
                        gr.Markdown("### 🔍 Arama & DSPy")
                        
                        enable_search = gr.Checkbox(
                            label="🌐 Web Araması",
                            value=True,
                            info="ⓘ DuckDuckGo ile güncel bilgi ara"
                        )
                        
                        enable_dspy = gr.Checkbox(
                            label="🧠 DSPy Zenginleştirme",
                            value=True,
                            info="ⓘ Otomatik mod seçimi ve prompt optimizasyonu"
                        )
                        
                        num_search_results = gr.Slider(
                            label="Arama Sonuç Sayısı",
                            minimum=1, maximum=10, value=5, step=1
                        )
                        
                        search_region = gr.Dropdown(
                            label="Arama Bölgesi",
                            choices=["tr-tr", "en-us", "de-de", "fr-fr", "wt-wt"],
                            value="tr-tr"
                        )
                    
                    with gr.Column(scale=1):
                        # LLM Parameters (visible in medium/expert mode)
                        gr.Markdown("### 📊 LLM Parametreleri")
                        
                        temperature = gr.Slider(
                            label="Temperature ⓘ",
                            minimum=0.0, maximum=2.0, value=0.7, step=0.05,
                            info="ⓘ Düşük (0.2) = odaklı, Yüksek (1.5) = yaratıcı"
                        )
                        
                        max_tokens = gr.Slider(
                            label="Max New Tokens ⓘ",
                            minimum=64, maximum=4096, value=512, step=64,
                            info="ⓘ Maksimum yanıt uzunluğu"
                        )
                        
                        top_p = gr.Slider(
                            label="Top-P ⓘ",
                            minimum=0.1, maximum=1.0, value=0.9, step=0.05,
                            info="ⓘ Nucleus sampling. Düşük = çeşitlilik az"
                        )
                        
                        top_k = gr.Slider(
                            label="Top-K ⓘ",
                            minimum=1, maximum=200, value=50, step=1,
                            info="ⓘ En iyi k token'dan örneklem"
                        )
                        
                        repetition_penalty = gr.Slider(
                            label="Repetition Penalty ⓘ",
                            minimum=1.0, maximum=2.0, value=1.1, step=0.05,
                            info="ⓘ Tekrarları önle. Yüksek = daha az tekrar"
                        )
                
                # System Prompt (visible in medium/expert mode)
                system_prompt_box = gr.Textbox(
                    label="📝 Sistem Prompt (Opsiyonel)",
                    placeholder="Sen bir yardımcı asistansın...",
                    lines=3,
                    info="ⓘ Modelin kişiliğini ve davranışını belirler"
                )
            
            # ═══════════════════════════════════════
            # STEP 3: CHAT
            # ═══════════════════════════════════════
            with gr.TabItem("3️⃣ Sohbet", id="step3"):
                gr.Markdown("### 💬 Sohbet")
                
                chatbot = gr.Chatbot(
                    label="Sohbet Geçmişi",
                    height=450,
                    avatar_images=(None, "https://cdn-icons-png.flaticon.com/512/4712/4712109.png")
                )
                
                pipeline_output = gr.Textbox(
                    label="📋 Detaylı Çıktı (Arama + DSPy + LLM)",
                    lines=8,
                    interactive=False,
                    elem_id="response-box"
                )
                
                with gr.Row():
                    prompt_input = gr.Textbox(
                        label="Prompt",
                        placeholder="Sorunuzu buraya yazın...",
                        lines=3,
                        scale=4
                    )
                    
                    with gr.Column(scale=1, min_width=120):
                        submit_btn = gr.Button("📤 Gönder", variant="primary", size="lg")
                        clear_chat_btn = gr.Button("🗑️ Temizle", size="sm")
                        new_session_btn = gr.Button("🔄 Yeni Session", size="sm")
                
                pipeline_status = gr.Textbox(
                    label="Durum",
                    value="Hazır.",
                    interactive=False,
                    max_lines=1
                )
        
        # ═══════════════════════════════════════
        # TABS: Logs, Gallery, Settings
        # ═══════════════════════════════════════
        with gr.Tabs():
            
            # Tab: Logs
            with gr.TabItem("📋 Loglar"):
                gr.Markdown("### 📋 İşlem Logları")
                
                with gr.Row():
                    log_type = gr.Dropdown(
                        label="Log Türü",
                        choices=["Özet", "Arama", "DSPy", "LLM", "Hata", "Genel", "Ham JSON"],
                        value="Özet"
                    )
                    session_only_cb = gr.Checkbox(label="Sadece bu session", value=True)
                    refresh_logs_btn = gr.Button("🔄 Yenile", variant="secondary")
                
                log_display = gr.Textbox(label="Log İçeriği", lines=25, interactive=False)
                
                gr.Markdown("---")
                gr.Markdown("### 🗑️ Log Temizleme")
                with gr.Row():
                    clear_table = gr.Dropdown(
                        label="Silinecek Tablo",
                        choices=["Tümü", "Arama", "DSPy", "LLM", "Hata", "Genel"],
                        value="Tümü"
                    )
                    clear_logs_btn = gr.Button("🗑️ Seçili Logları Sil", variant="stop")
                clear_status = gr.Textbox(label="Durum", interactive=False, max_lines=1)
            
            # Tab: Model Gallery
            with gr.TabItem("🛍️ Model Galerisi"):
                gr.Markdown("### 🛍️ Model Galerisi — Canlı HuggingFace Kataloğu")
                
                with gr.Tabs():
                    with gr.TabItem("⚡ OpenVINO"):
                        ov_search = gr.Textbox(label="🔎 Ara", placeholder="qwen, mistral...", scale=3)
                        ov_refresh_btn = gr.Button("🔄 Yenile", scale=1)
                        ov_catalog_table = gr.Dataframe(
                            headers=["Model Adı", "HF ID", "Boyut", "Context", "İndirme", "Etiketler"],
                            interactive=False, wrap=True
                        )
                        with gr.Row():
                            ov_catalog_select = gr.Dropdown(label="Seç", choices=[], scale=3)
                            ov_catalog_dl_btn = gr.Button("⬇️ İndir", scale=1)
                            ov_catalog_load_btn = gr.Button("⚡ Yükle", scale=1)
                        ov_catalog_status = gr.Textbox(label="Durum", lines=2)
                    
                    with gr.TabItem("🦙 Ollama"):
                        ollama_search = gr.Textbox(label="🔎 Ara", placeholder="qwen, mistral...")
                        ollama_refresh_btn = gr.Button("🔄 Yenile")
                        ollama_catalog_table = gr.Dataframe(
                            headers=["Model Adı", "Ollama ID", "Boyut", "Context", "Etiketler"],
                            interactive=False, wrap=True
                        )
                        with gr.Row():
                            ollama_catalog_select = gr.Dropdown(label="Seç", choices=[], scale=3)
                            ollama_catalog_pull_btn = gr.Button("⬇️ İndir", scale=1)
                            ollama_catalog_load_btn = gr.Button("⚡ Yükle", scale=1)
                        ollama_catalog_status = gr.Textbox(label="Durum", lines=2)
                    
                    with gr.TabItem("🔷 GGUF"):
                        gguf_search = gr.Textbox(label="🔎 Ara", placeholder="qwen, mistral...")
                        gguf_refresh_btn = gr.Button("🔄 Yenile")
                        gguf_catalog_table = gr.Dataframe(
                            headers=["Model Adı", "Repo", "Boyut", "Context", "İndirme", "Etiketler"],
                            interactive=False, wrap=True
                        )
                        with gr.Row():
                            gguf_catalog_select = gr.Dropdown(label="Seç", choices=[], scale=3)
                            gguf_catalog_dl_btn = gr.Button("⬇️ İndir", scale=1)
                            gguf_catalog_load_btn = gr.Button("⚡ Yükle", scale=1)
                        gguf_catalog_status = gr.Textbox(label="Durum", lines=2)
            
            # Tab: Settings
            with gr.TabItem("⚙️ Ayarlar"):
                gr.Markdown("### ⚙️ Uygulama Ayarları")
                
                gr.Markdown("#### 🎯 Kullanıcı Modu")
                settings_mode_selector = gr.Radio(
                    label="UI Karmaşıklık Seviyesi",
                    choices=["🔰 Basit", "⚙️ Orta", "🔬 Expert"],
                    value="⚙️ Orta",
                    info="Basit: Sadece temel özellikler | Orta: Önerilen | Expert: Tüm özellikler"
                )
                
                gr.Markdown("---")
                gr.Markdown("#### 📊 Sistem İstatistikleri")
                stats_display = gr.Textbox(label="Sistem Durumu", lines=12, interactive=False)
                refresh_stats_btn = gr.Button("🔄 Yenile", variant="secondary")
        
        # ═══════════════════════════════════════
        # EVENT HANDLERS
        # ═══════════════════════════════════════
        
        # Mode selector changes UI visibility
        mode_selector.change(
            fn=update_ui_for_mode,
            inputs=[mode_selector],
            outputs=[gpu_accordion, temperature, system_prompt_box]
        )
        
        # Preset applies to LLM parameters
        preset_selector.change(
            fn=apply_preset,
            inputs=[preset_selector],
            outputs=[temperature, max_tokens, top_p, top_k, repetition_penalty]
        )
        
        # Backend status refresh
        refresh_backend_btn.click(
            fn=lambda: get_backend_status_display(),
            outputs=[backend_status_md]
        )
        
        # Model refresh
        refresh_models_btn.click(
            fn=refresh_models,
            outputs=[model_dropdown, load_status]
        )
        
        # Model load
        load_model_btn.click(
            fn=load_model_action,
            inputs=[model_dropdown, device_radio, gpu_max_alloc, kv_cache_prec,
                    num_streams, perf_hint, cache_dir],
            outputs=[load_status, submit_btn]
        )
        
        # Ollama pull
        ollama_pull_btn.click(
            fn=lambda x: ("❌ Henüz implement edilmedi", ""),
            inputs=[ollama_pull_input],
            outputs=[ollama_pull_status]
        )
        
        # Chat submission
        submit_btn.click(
            fn=run_inference,
            inputs=[prompt_input, system_prompt_box, enable_search, enable_dspy,
                    num_search_results, search_region, temperature, max_tokens,
                    top_p, top_k, repetition_penalty, chatbot],
            outputs=[chatbot, pipeline_output, status_html]
        )
        
        prompt_input.submit(
            fn=run_inference,
            inputs=[prompt_input, system_prompt_box, enable_search, enable_dspy,
                    num_search_results, search_region, temperature, max_tokens,
                    top_p, top_k, repetition_penalty, chatbot],
            outputs=[chatbot, pipeline_output, status_html]
        )
        
        # Clear chat
        clear_chat_btn.click(
            fn=lambda: ([], "", "Hazır."),
            outputs=[chatbot, pipeline_output, pipeline_status]
        )
        
        # New session
        new_session_btn.click(
            fn=lambda: (_get_orch().new_session(), [], "", "Yeni session başlatıldı."),
            outputs=[status_html, chatbot, pipeline_output, pipeline_status]
        )
        
        # Log refresh
        refresh_logs_btn.click(
            fn=lambda: "Log fonksiyonu yakında eklenecek",
            outputs=[log_display]
        )
        
        # Clear logs
        clear_logs_btn.click(
            fn=lambda: "Log temizleme fonksiyonu yakında eklenecek",
            outputs=[clear_status]
        )
        
        # Stats refresh
        refresh_stats_btn.click(
            fn=lambda: f"Sistem İstatistikleri\n\nCPU: {get_system_status()['cpu']:.1f}%\nRAM: {get_system_status()['memory']:.1f}%\nModel: {get_system_status()['model_name']}\nBackend: {get_system_status()['backend'].upper()}",
            outputs=[stats_display]
        )
        
        # Auto-refresh status with Timer (Gradio 6.0)
        timer = gr.Timer(value=10)
        timer.tick(
            fn=lambda: format_status_html(get_system_status()),
            outputs=[status_html]
        )
    
    return demo


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("OpenVINO LLM Studio Modern UI başlatılıyor...")
    
    demo = build_modern_ui()
    
    # Port selection
    host = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    env_port = os.getenv("GRADIO_SERVER_PORT")
    
    if env_port:
        server_port = int(env_port)
    else:
        # Auto-select port in range 7860-7870
        server_port = 7860
        for p in range(7860, 7871):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    s.bind((host, p))
                    server_port = p
                    break
            except OSError:
                continue
    
    logger.info(f"Gradio server: {host}:{server_port}")
    
    demo.launch(
        server_name=host,
        server_port=server_port,
        show_error=True,
        inbrowser=True,
        share=False,
        css=CSS,
    )
