"""
ui/app_unified.py
Birleştirilmiş OpenVINO LLM Studio UI.

Tek bir UI'da üç farklı görünüm modu:
- Klasik: Tek ekran, tüm ayarlar görünür
- Modern: 3 adımlı workflow (Model → Ayarlar → Chat)
- Workspace: Sidebar + Ana Kanvas düzeni

Her mod kendi state'ini StateManager üzerinden yönetir.
"""

import os
import sys
import json
import logging
import threading
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import gradio as gr
from core.orchestrator import Orchestrator
from core.config import OPENVINO_LLM_HOME, CACHE_DIR, GGUF_DIR, LOG_DIR

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "studio_unified.log", encoding="utf-8"),
    ]
)
logger = logging.getLogger(__name__)

_orch_lock = threading.Lock()
_orch = None

def _get_orch() -> Orchestrator:
    """Unified UI için orchestrator instance (ui_id='unified')."""
    global _orch
    with _orch_lock:
        if _orch is None:
            _orch = Orchestrator(ui_id="unified")
        return _orch


# ═══════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

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
        "memory_gb": mem.used / (1024**3),
        "memory_total": mem.total / (1024**3),
        "model_loaded": model_loaded,
        "model_name": model_name,
        "backend": orch._active_backend,
    }


def format_footer_html(status: dict) -> str:
    """Format footer status bar."""
    model_status = (
        f'<span style="color:#22c55e">✅ {status["model_name"]}</span>'
        if status["model_loaded"] else
        '<span style="color:#94a3b8">Model Yok</span>'
    )

    cpu_color = "#22c55e" if status["cpu"] < 50 else "#f59e0b" if status["cpu"] < 80 else "#ef4444"
    mem_color = "#22c55e" if status["memory"] < 50 else "#f59e0b" if status["memory"] < 80 else "#ef4444"

    return f'''
    <div style="display: flex; justify-content: space-between; align-items: center; padding: 6px 12px; background: #f1f5f9; border-top: 1px solid #e2e8f0; font-family: monospace; font-size: 0.8em;">
        <div style="display: flex; gap: 16px;">
            <span><b>⚡ Backend:</b> {status["backend"].upper()}</span>
            <span><b>🤖 Model:</b> {model_status}</span>
        </div>
        <div style="display: flex; gap: 16px;">
            <span><b>💻 CPU:</b> <span style="color:{cpu_color}">{status["cpu"]:.1f}%</span></span>
            <span><b>💾 RAM:</b> <span style="color:{mem_color}">{status["memory_gb"]:.1f}/{status["memory_total"]:.0f}GB ({status["memory"]:.0f}%)</span></span>
        </div>
    </div>
    '''


def refresh_models():
    """Refresh model list."""
    orch = _get_orch()
    backend = orch._active_backend
    choices = orch.get_model_choices()

    if choices:
        return gr.update(choices=choices, value=choices[0]), f"✅ {len(choices)} model ({backend})"

    if backend == "openvino":
        msg = "⚠️ OpenVINO model yok — Model Galerisi'nden indirin"
    elif backend == "ollama":
        msg = "⚠️ Ollama model yok — 'ollama pull <model>' ile indirin"
    elif backend == "ipex":
        msg = "⚠️ GGUF model yok — Model Galerisi'nden indirin"
    else:
        msg = "⚠️ Model yok"

    return gr.update(choices=[], value=None), msg


def load_model_action(model_choice, device, gpu_max_alloc, kv_cache_prec, num_streams, perf_hint, cache_dir):
    """Load model action."""
    orch = _get_orch()
    if not model_choice:
        return "❌ Model seçilmedi.", gr.update()

    backend = orch._active_backend
    
    # Ollama için model ön yükleme
    if backend == "ollama":
        try:
            import requests
            # Ollama model yükleme (keep_alive=-1 ile bellekte tut)
            r = requests.post(
                f"{orch.ollama.base_url}/api/generate",
                json={"model": model_choice, "prompt": "", "keep_alive": -1},
                timeout=300  # 5 dakika timeout
            )
            if r.status_code == 200:
                # _current_model set et - bu önemli!
                orch.ollama._current_model = model_choice
                status = get_system_status()
                return f"✅ Ollama modeli '{model_choice}' yüklendi ve bellekte tutuluyor.", gr.update(value=format_footer_html(status))
        except requests.Timeout:
            return "⏳ Model yükleniyor... (bu işlem birkaç dakika sürebilir)", gr.update()
        except Exception as e:
            return f"❌ Hata: {e}", gr.update()
    
    # OpenVINO ve IPEX için normal yükleme
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
        if cache_dir.strip():
            ov_config["CACHE_DIR"] = cache_dir.strip()

    success, msg = orch.load_model(model_choice, device, ov_config=ov_config)
    status = get_system_status()
    return msg, gr.update(value=format_footer_html(status))


def run_inference(prompt, enable_search, enable_dspy, num_search_results,
                  temperature, max_tokens, top_p, top_k, repetition_penalty,
                  system_prompt, history):
    """Run inference pipeline."""
    orch = _get_orch()
    backend = orch._active_backend

    if not prompt.strip():
        yield history, "", gr.update()
        return

    # Ollama için model yükleme bekleme - Ollama otomatik yönetir
    if backend == "ollama":
        # Ollama'da model yoksa ilk request'te yüklenir, timeout'a takılmayız
        if not orch.is_model_loaded:
            yield history, "⏳ Ollama modeli başlatılıyor...", gr.update()
    else:
        # OpenVINO ve IPEX için model yüklü olmalı
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

    chat_history = []
    for turn in (history or []):
        if isinstance(turn, dict):
            chat_history.append(turn)

    for chunk in orch.run_pipeline(
        prompt=prompt,
        params=params,
        enable_search=enable_search,
        enable_dspy=enable_dspy,
        num_search_results=int(num_search_results),
        system_prompt=system_prompt,
        history=chat_history if chat_history else None,
    ):
        accumulated += chunk
        yield history, accumulated, gr.update()

    history = list(history)
    history.append({"role": "user", "content": prompt})
    history.append({"role": "assistant", "content": accumulated})
    yield history, accumulated, gr.update(value=format_footer_html(get_system_status()))


def get_logs_display(log_type, session_only):
    """Display logs in UI."""
    orch = _get_orch()
    logs = orch.get_logs(session_only=session_only)

    if log_type == "Ham JSON":
        return json.dumps(logs, ensure_ascii=False, indent=2, default=str)

    lines = []
    if log_type in ("Özet", "Arama"):
        searches = logs.get("search", [])
        if searches:
            lines.append(f"🔍 ARAMA LOGLARI ({len(searches)} kayıt)\n" + "━"*50)
            for r in searches[-10:]:
                lines.append(f"📅 {r.get('timestamp','')}")
                lines.append(f"   Prompt: {r.get('original_prompt','')[:80]}")
                lines.append(f"   Sorgu: {r.get('search_query','')}")
                lines.append(f"   Sonuç: {r.get('num_results',0)} sonuç | {r.get('duration_ms',0):.0f}ms")
                lines.append("")

    if log_type in ("Özet", "DSPy"):
        dspy_logs = logs.get("dspy", [])
        if dspy_logs:
            lines.append(f"🧠 DSPy LOGLARI ({len(dspy_logs)} kayıt)\n" + "━"*50)
            for r in dspy_logs[-10:]:
                lines.append(f"📅 {r.get('timestamp','')}")
                lines.append(f"   Mod: {r.get('detected_mode','')} | Sebep: {r.get('mode_reason','')}")
                lines.append(f"   Süre: {r.get('duration_ms',0):.0f}ms")
                lines.append("")

    if log_type in ("Özet", "LLM"):
        llm_logs = logs.get("llm", [])
        if llm_logs:
            lines.append(f"⚡ LLM LOGLARI ({len(llm_logs)} kayıt)\n" + "━"*50)
            for r in llm_logs[-10:]:
                lines.append(f"📅 {r.get('timestamp','')} | {r.get('model_name','')}")
                lines.append(f"   Input→Output: {r.get('input_tokens',0)}→{r.get('output_tokens',0)} token")
                lines.append(f"   Performans: {r.get('tokens_per_second',0):.1f} tok/s")
                lines.append("")

    if log_type in ("Özet", "Hata"):
        errors = logs.get("errors", [])
        if errors:
            lines.append(f"❌ HATA LOGLARI ({len(errors)} kayıt)\n" + "━"*50)
            for r in errors[-10:]:
                lines.append(f"📅 {r.get('timestamp','')} | {r.get('module','')}")
                lines.append(f"   Tip: {r.get('error_type','?')}")
                lines.append(f"   Mesaj: {r.get('error_message','')}")
                lines.append("")

    return "\n".join(lines) if lines else "Henüz log kaydı yok."


def clear_logs_action():
    """Clear logs."""
    orch = _get_orch()
    success = orch.clear_logs("all")
    return "✅ Loglar temizlendi." if success else "❌ Log temizleme başarısız."


def apply_preset(preset):
    """Apply preset configuration."""
    presets = {
        "🚀 Hızlı": (0.5, 256, 0.8, 40, 1.05),
        "⚖️ Dengeli": (0.7, 512, 0.9, 50, 1.1),
        "🎨 Kaliteli": (0.9, 1024, 0.95, 100, 1.15),
    }
    if preset in presets:
        t, m, p, k, r = presets[preset]
        return [gr.update(value=v) for v in [t, m, p, k, r]]
    return [gr.update() for _ in range(5)]


def on_backend_change(b):
    _get_orch().set_backend(b)
    models, status = refresh_models()
    return models


def on_model_select(model_choice, backend):
    """Model seçildiğinde _current_model set et (Ollama için)."""
    if backend == "ollama" and model_choice:
        orch = _get_orch()
        orch.ollama._current_model = model_choice
    return gr.update()


# ═══════════════════════════════════════════════════════════════════
# CSS & THEME
# ═══════════════════════════════════════════════════════════════════

CSS = """
#header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    padding: 16px 24px;
    border-bottom: 1px solid #334155;
    margin-bottom: 0;
}
#header h1 { color: #e2e8f0; font-size: 1.5em; margin: 0; font-weight: 600; }
#header p { color: #94a3b8; margin: 4px 0 0 0; font-size: 0.9em; }

.footer-status {
    position: fixed;
    bottom: 0; left: 0; right: 0;
    z-index: 100;
}

.sidebar-panel {
    background: #f8fafc;
    border-right: 1px solid #e2e8f0;
    padding: 12px;
}

#chatbot-unified { height: 550px !important; max-height: 65vh; }

.accordions-wrapper .accordion {
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    margin-bottom: 8px;
}

/* Classic UI specific */
.classic-tabs .tab-nav {
    background: #f1f5f9;
    padding: 8px;
    border-radius: 8px;
}

/* Modern UI specific */
.step-container {
    background: #f9fafb;
    padding: 16px;
    border-radius: 8px;
    margin: 8px 0;
}
"""

THEME = gr.themes.Base(primary_hue="slate", neutral_hue="gray")


# ═══════════════════════════════════════════════════════════════════
# UI COMPONENT BUILDERS
# ═══════════════════════════════════════════════════════════════════

def build_model_settings():
    """Build model settings accordion."""
    with gr.Accordion("🤖 Model Seçimi", open=True):
        backend_radio = gr.Radio(
            label="Backend",
            choices=["openvino", "ollama", "ipex"],
            value="openvino",
        )
        with gr.Row():
            model_dropdown = gr.Dropdown(
                label="Model", choices=[], interactive=True, allow_custom_value=True
            )
            refresh_btn = gr.Button("🔄", size="sm", scale=0)
        device_radio = gr.Radio(
            label="Cihaz", choices=["CPU", "GPU", "AUTO"], value="CPU"
        )
        load_btn = gr.Button("⚡ Modeli Yükle", variant="primary", size="md")
        load_status = gr.Textbox(label="Durum", value="Model yüklenmedi.", interactive=False, max_lines=2)
    return backend_radio, model_dropdown, refresh_btn, device_radio, load_btn, load_status


def build_gpu_settings():
    """Build GPU settings accordion."""
    with gr.Accordion("⚙️ GPU/CPU Ayarları", open=False):
        gpu_max_alloc = gr.Slider(label="GPU Max Alloc %", minimum=40, maximum=95, value=75, step=5)
        kv_cache_prec = gr.Radio(label="KV Cache Precision", choices=["f16", "u8"], value="u8")
        num_streams = gr.Slider(label="Num Streams", minimum=1, maximum=4, value=1, step=1)
        perf_hint = gr.Radio(label="Performance Hint", choices=["LATENCY", "THROUGHPUT"], value="LATENCY")
        cache_dir = gr.Textbox(label="Cache Dizini", value=str(CACHE_DIR))
    return gpu_max_alloc, kv_cache_prec, num_streams, perf_hint, cache_dir


def build_llm_params():
    """Build LLM parameters accordion."""
    with gr.Accordion("📊 LLM Parametreleri", open=True):
        preset_selector = gr.Radio(
            label="Preset", choices=["🚀 Hızlı", "⚖️ Dengeli", "🎨 Kaliteli"], value="⚖️ Dengeli"
        )
        temperature = gr.Slider(label="Temperature", minimum=0.0, maximum=2.0, value=0.7, step=0.05)
        max_tokens = gr.Slider(label="Max Tokens", minimum=64, maximum=4096, value=512, step=64)
        top_p = gr.Slider(label="Top-P", minimum=0.1, maximum=1.0, value=0.9, step=0.05)
        top_k = gr.Slider(label="Top-K", minimum=1, maximum=200, value=50, step=1)
        repetition_penalty = gr.Slider(label="Repetition Penalty", minimum=1.0, maximum=2.0, value=1.1, step=0.05)
    return preset_selector, temperature, max_tokens, top_p, top_k, repetition_penalty


def build_search_dspy():
    """Build Search & DSPy accordion."""
    with gr.Accordion("🔍 Arama & DSPy", open=True):
        enable_search = gr.Checkbox(label="🌐 Web Araması", value=True)
        enable_dspy = gr.Checkbox(label="🧠 DSPy", value=True)
        num_search_results = gr.Slider(label="Sonuç Sayısı", minimum=1, maximum=10, value=5, step=1)
    return enable_search, enable_dspy, num_search_results


def build_system_prompt():
    """Build System Prompt accordion."""
    with gr.Accordion("📝 Sistem Prompt", open=False):
        system_prompt = gr.Textbox(label="Sistem Prompt", placeholder="Sen bir yardımcı asistansın...", lines=3)
    return system_prompt


def build_chat_area():
    """Build chat area."""
    chatbot = gr.Chatbot(
        label="Sohbet", height=550, elem_id="chatbot-unified",
        avatar_images=(None, "https://cdn-icons-png.flaticon.com/512/4712/4712109.png")
    )
    with gr.Accordion("📋 Detaylı Çıktı", open=False):
        pipeline_output = gr.Textbox(label="Detaylı Çıktı", lines=6, interactive=False)
    with gr.Row():
        prompt_input = gr.Textbox(label="Prompt", placeholder="Sorunuzu buraya yazın...", lines=3, scale=4, show_label=False)
        with gr.Column(scale=1, min_width=120):
            submit_btn = gr.Button("📤 Gönder", variant="primary", size="lg")
            clear_btn = gr.Button("🗑️ Temizle", size="sm")
    return chatbot, pipeline_output, prompt_input, submit_btn, clear_btn


def build_logs_panel():
    """Build logs panel."""
    with gr.Accordion("📋 Loglar", open=False):
        with gr.Row():
            log_type = gr.Dropdown(
                label="Log Türü",
                choices=["Özet", "Arama", "DSPy", "LLM", "Hata", "Genel", "Ham JSON"],
                value="Özet",
            )
            refresh_logs_btn = gr.Button("🔄 Yenile", size="sm")
            clear_logs_btn = gr.Button("🗑️ Temizle", size="sm", variant="stop")
        log_display = gr.Textbox(label="Log İçeriği", lines=15, interactive=False)
        clear_logs_status = gr.Textbox(label="Durum", interactive=False, max_lines=1, visible=False)
    return log_type, refresh_logs_btn, clear_logs_btn, log_display, clear_logs_status


# ═══════════════════════════════════════════════════════════════════
# MAIN UI BUILDER - THREE MODES
# ═══════════════════════════════════════════════════════════════════

def build_unified_ui():
    """Build unified UI with three distinct modes."""

    with gr.Blocks(title="OpenVINO LLM Studio - Unified", css=CSS) as demo:

        # Header
        with gr.Row(elem_id="header"):
            with gr.Column(scale=1):
                gr.HTML("<h1>⚡ OpenVINO LLM Studio</h1>")
                gr.HTML("<p>Unified UI · 3 Mod Seçeneği</p>")
            with gr.Column(scale=0, min_width=300):
                ui_mode = gr.Radio(
                    label="UI Modu",
                    choices=["🏠 Klasik", "✨ Modern", "💼 Workspace"],
                    value="🏠 Klasik",
                    show_label=False,
                    container=False,
                )

        # Footer status bar
        status_html = gr.HTML(value=format_footer_html(get_system_status()), elem_classes=["footer-status"])

        # ═══════════════════════════════════════
        # MODE 1: KLASIK UI (Tek ekran, Tabs)
        # ═══════════════════════════════════════
        with gr.Column(visible=True) as classic_mode:
            with gr.Tabs(elem_classes=["classic-tabs"]):
                with gr.TabItem("💬 Chat"):
                    with gr.Row():
                        with gr.Column(scale=1, min_width=300):
                            backend_radio_c, model_dropdown_c, refresh_btn_c, device_radio_c, load_btn_c, load_status_c = build_model_settings()
                            build_gpu_settings()
                            build_search_dspy()
                        with gr.Column(scale=2):
                            chatbot_c, pipeline_output_c, prompt_input_c, submit_btn_c, clear_btn_c = build_chat_area()
                
                with gr.TabItem("⚙️ Ayarlar"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            preset_selector_c, temperature_c, max_tokens_c, top_p_c, top_k_c, repetition_penalty_c = build_llm_params()
                        with gr.Column(scale=1):
                            build_system_prompt()
                
                with gr.TabItem("📋 Loglar"):
                    log_type_c, refresh_logs_c, clear_logs_c, log_display_c, clear_logs_status_c = build_logs_panel()

        # ═══════════════════════════════════════
        # MODE 2: MODERN UI (3 adımlı workflow)
        # ═══════════════════════════════════════
        with gr.Column(visible=False) as modern_mode:
            gr.Markdown("### 1️⃣ Model Seçimi", elem_classes=["step-container"])
            with gr.Row():
                with gr.Column(scale=1):
                    backend_radio_m, model_dropdown_m, refresh_btn_m, device_radio_m, load_btn_m, load_status_m = build_model_settings()
                    build_gpu_settings()
                with gr.Column(scale=2):
                    gr.Markdown("*Model seçin ve yükleyin, sonra Ayarlar sekmesine geçin.*")
            
            gr.Markdown("### 2️⃣ Ayarlar", elem_classes=["step-container"])
            with gr.Row():
                with gr.Column(scale=1):
                    build_search_dspy()
                with gr.Column(scale=1):
                    preset_selector_m, temperature_m, max_tokens_m, top_p_m, top_k_m, repetition_penalty_m = build_llm_params()
            build_system_prompt()
            
            gr.Markdown("### 3️⃣ Sohbet", elem_classes=["step-container"])
            chatbot_m, pipeline_output_m, prompt_input_m, submit_btn_m, clear_btn_m = build_chat_area()
            
            gr.Markdown("### 📋 Loglar", elem_classes=["step-container"])
            log_type_m, refresh_logs_m, clear_logs_m, log_display_m, clear_logs_status_m = build_logs_panel()

        # ═══════════════════════════════════════
        # MODE 3: WORKSPACE UI (Sidebar + Kanvas)
        # ═══════════════════════════════════════
        with gr.Column(visible=False) as workspace_mode:
            with gr.Row(equal_height=False):
                # Left Sidebar
                with gr.Column(scale=1, min_width=320, elem_classes=["sidebar-panel"]):
                    backend_radio_w, model_dropdown_w, refresh_btn_w, device_radio_w, load_btn_w, load_status_w = build_model_settings()
                    build_gpu_settings()
                    preset_selector_w, temperature_w, max_tokens_w, top_p_w, top_k_w, repetition_penalty_w = build_llm_params()
                    build_search_dspy()
                    build_system_prompt()
                
                # Right Canvas
                with gr.Column(scale=3, min_width=500):
                    chatbot_w, pipeline_output_w, prompt_input_w, submit_btn_w, clear_btn_w = build_chat_area()
                    log_type_w, refresh_logs_w, clear_logs_w, log_display_w, clear_logs_status_w = build_logs_panel()

        # ═══════════════════════════════════════
        # EVENT HANDLERS
        # ═══════════════════════════════════════

        # UI mode switcher
        def switch_mode(mode):
            return [
                gr.update(visible=(mode == "🏠 Klasik")),
                gr.update(visible=(mode == "✨ Modern")),
                gr.update(visible=(mode == "💼 Workspace")),
            ]

        ui_mode.change(
            fn=switch_mode,
            inputs=[ui_mode],
            outputs=[classic_mode, modern_mode, workspace_mode],
        )

        # Preset selector (all modes)
        preset_selector_c.change(fn=apply_preset, inputs=[preset_selector_c], outputs=[temperature_c, max_tokens_c, top_p_c, top_k_c, repetition_penalty_c])
        preset_selector_m.change(fn=apply_preset, inputs=[preset_selector_m], outputs=[temperature_m, max_tokens_m, top_p_m, top_k_m, repetition_penalty_m])
        preset_selector_w.change(fn=apply_preset, inputs=[preset_selector_w], outputs=[temperature_w, max_tokens_w, top_p_w, top_k_w, repetition_penalty_w])

        # Backend change (all modes)
        backend_radio_c.change(fn=on_backend_change, inputs=[backend_radio_c], outputs=[model_dropdown_c])
        backend_radio_m.change(fn=on_backend_change, inputs=[backend_radio_m], outputs=[model_dropdown_m])
        backend_radio_w.change(fn=on_backend_change, inputs=[backend_radio_w], outputs=[model_dropdown_w])
        
        # Model select (Ollama için _current_model set et)
        model_dropdown_c.change(fn=on_model_select, inputs=[model_dropdown_c, backend_radio_c], outputs=[])
        model_dropdown_m.change(fn=on_model_select, inputs=[model_dropdown_m, backend_radio_m], outputs=[])
        model_dropdown_w.change(fn=on_model_select, inputs=[model_dropdown_w, backend_radio_w], outputs=[])

        # Refresh models (all modes)
        refresh_btn_c.click(fn=refresh_models, outputs=[model_dropdown_c, load_status_c])
        refresh_btn_m.click(fn=refresh_models, outputs=[model_dropdown_m, load_status_m])
        refresh_btn_w.click(fn=refresh_models, outputs=[model_dropdown_w, load_status_w])

        # Load model (all modes)
        load_btn_c.click(fn=load_model_action, inputs=[model_dropdown_c, device_radio_c, gr.State(75), gr.State("u8"), gr.State(1), gr.State("LATENCY"), gr.State(str(CACHE_DIR))], outputs=[load_status_c, status_html])
        load_btn_m.click(fn=load_model_action, inputs=[model_dropdown_m, device_radio_m, gr.State(75), gr.State("u8"), gr.State(1), gr.State("LATENCY"), gr.State(str(CACHE_DIR))], outputs=[load_status_m, status_html])
        load_btn_w.click(fn=load_model_action, inputs=[model_dropdown_w, device_radio_w, gr.State(75), gr.State("u8"), gr.State(1), gr.State("LATENCY"), gr.State(str(CACHE_DIR))], outputs=[load_status_w, status_html])

        # Submit (all modes)
        submit_btn_c.click(fn=run_inference, inputs=[prompt_input_c, gr.State(True), gr.State(True), gr.State(5), temperature_c, max_tokens_c, top_p_c, top_k_c, repetition_penalty_c, gr.State(""), chatbot_c], outputs=[chatbot_c, pipeline_output_c, status_html])
        submit_btn_m.click(fn=run_inference, inputs=[prompt_input_m, gr.State(True), gr.State(True), gr.State(5), temperature_m, max_tokens_m, top_p_m, top_k_m, repetition_penalty_m, gr.State(""), chatbot_m], outputs=[chatbot_m, pipeline_output_m, status_html])
        submit_btn_w.click(fn=run_inference, inputs=[prompt_input_w, gr.State(True), gr.State(True), gr.State(5), temperature_w, max_tokens_w, top_p_w, top_k_w, repetition_penalty_w, gr.State(""), chatbot_w], outputs=[chatbot_w, pipeline_output_w, status_html])

        # Clear chat (all modes)
        clear_btn_c.click(fn=lambda: ([], ""), outputs=[chatbot_c, pipeline_output_c])
        clear_btn_m.click(fn=lambda: ([], ""), outputs=[chatbot_m, pipeline_output_m])
        clear_btn_w.click(fn=lambda: ([], ""), outputs=[chatbot_w, pipeline_output_w])

        # Logs (all modes)
        refresh_logs_c.click(fn=lambda lt: get_logs_display(lt, True), inputs=[log_type_c], outputs=[log_display_c])
        refresh_logs_m.click(fn=lambda lt: get_logs_display(lt, True), inputs=[log_type_m], outputs=[log_display_m])
        refresh_logs_w.click(fn=lambda lt: get_logs_display(lt, True), inputs=[log_type_w], outputs=[log_display_w])
        clear_logs_c.click(fn=clear_logs_action, outputs=[clear_logs_status_c])
        clear_logs_m.click(fn=clear_logs_action, outputs=[clear_logs_status_m])
        clear_logs_w.click(fn=clear_logs_action, outputs=[clear_logs_status_w])

        # Initial load
        demo.load(fn=refresh_models, outputs=[model_dropdown_c, load_status_c])
        demo.load(fn=refresh_models, outputs=[model_dropdown_m, load_status_m])
        demo.load(fn=refresh_models, outputs=[model_dropdown_w, load_status_w])

    return demo


if __name__ == "__main__":
    import os
    # Gradio timeout ayarları - büyük modeller için 5 dakika
    os.environ["GRADIO_SERVER_TIMEOUT"] = "300"
    os.environ["GRADIO_TEMP_DIR"] = str(LOG_DIR / "gradio_tmp")
    
    demo = build_unified_ui()
    # Queue ayarları
    demo.queue(default_concurrency_limit=10).launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        theme=THEME,
    )
