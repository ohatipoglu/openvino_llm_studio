"""
ui/app_workspace.py

Modern Workspace UI - Sidebar + Ana Kanvas düzeni
- Sürekli erişilebilir ayarlar (Sidebar)
- Merkezde sohbet (Ana Kanvas)
- Minimal header, bilgilendirici footer
- Accordion'lar ile organize edilmiş ayarlar
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
        logging.FileHandler(LOG_DIR / "studio_workspace.log", encoding="utf-8"),
    ]
)
logger = logging.getLogger(__name__)

_orch_lock = threading.Lock()
_orch = None

def _get_orch() -> Orchestrator:
    """Workspace UI için orchestrator instance (ui_id='workspace')."""
    global _orch
    with _orch_lock:
        if _orch is None:
            _orch = Orchestrator(ui_id="workspace")
        return _orch


# ═══════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def get_system_status():
    """Get system status for footer."""
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
    
    logger.info(f"Refresh models: backend={backend}, found={len(choices) if choices else 0} models")
    
    if choices:
        return gr.update(choices=choices, value=choices[0]), f"✅ {len(choices)} model ({backend})"
    
    # Backend-specific message
    if backend == "openvino":
        msg = "⚠️ OpenVINO model yok — Model Galerisi'nden indirin veya C:\\OpenVINO_LLM\\ dizinine yerleştirin"
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

    if not prompt.strip():
        yield history, "", gr.update()
        return

    if not orch.is_model_loaded:
        yield history, "❌ Model yüklü değil.", gr.update()
        return

    params = {
        "temperature": temperature,
        "max_tokens": int(max_tokens),
        "top_p": top_p,
        "top_k": int(top_k),
        "repetition_penalty": repetition_penalty,
    }

    accumulated = ""

    # Chatbot geçmişini backend'e uygun formata dönüştür
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

    # Search Logs
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
        elif log_type == "Arama":
            lines.append("Arama logu bulunamadı.")

    # DSPy Logs
    if log_type in ("Özet", "DSPy"):
        dspy_logs = logs.get("dspy", [])
        if dspy_logs:
            lines.append(f"🧠 DSPy LOGLARI ({len(dspy_logs)} kayıt)\n" + "━"*50)
            for r in dspy_logs[-10:]:
                lines.append(f"📅 {r.get('timestamp','')}")
                lines.append(f"   Mod: {r.get('detected_mode','')} | Sebep: {r.get('mode_reason','')}")
                lines.append(f"   Süre: {r.get('duration_ms',0):.0f}ms")
                lines.append(f"   Prompt (ilk 100): {str(r.get('original_prompt',''))[:100]}")
                lines.append("")
        elif log_type == "DSPy":
            lines.append("DSPy logu bulunamadı.")

    # LLM Logs
    if log_type in ("Özet", "LLM"):
        llm_logs = logs.get("llm", [])
        if llm_logs:
            lines.append(f"⚡ LLM LOGLARI ({len(llm_logs)} kayıt)\n" + "━"*50)
            for r in llm_logs[-10:]:
                lines.append(f"📅 {r.get('timestamp','')} | {r.get('model_name','')}")
                lines.append(f"   Backend: {r.get('backend','?')} | Input→Output: {r.get('input_tokens',0)}→{r.get('output_tokens',0)} token")
                lines.append(f"   Performans: {r.get('tokens_per_second',0):.1f} tok/s | {r.get('duration_ms',0):.0f}ms")
                if r.get('response'):
                    lines.append(f"   Yanıt (ilk 200): {str(r.get('response',''))[:200].replace(chr(10),' | ')}")
                lines.append("")
        elif log_type == "LLM":
            lines.append("LLM logu bulunamadı.")

    # Error Logs
    if log_type in ("Özet", "Hata"):
        errors = logs.get("errors", [])
        if errors:
            lines.append(f"❌ HATA LOGLARI ({len(errors)} kayıt)\n" + "━"*50)
            for r in errors[-10:]:
                lines.append(f"📅 {r.get('timestamp','')} | {r.get('module','')}")
                lines.append(f"   Tip: {r.get('error_type','?')}")
                lines.append(f"   Mesaj: {r.get('error_message','')}")
                lines.append("")
        elif log_type == "Hata":
            lines.append("Hata logu bulunamadı.")

    # General Logs
    if log_type in ("Özet", "Genel"):
        general = logs.get("general", [])
        if general:
            lines.append(f"📝 GENEL LOGLAR ({len(general)} kayıt)\n" + "━"*50)
            for r in general[-20:]:
                icon = {"INFO": "ℹ️", "WARNING": "⚠️", "ERROR": "❌", "DEBUG": "🔧"}.get(r.get("level",""), "•")
                lines.append(f"{icon} {r.get('timestamp','')} [{r.get('module','')}] {r.get('message','')[:100]}")
        elif log_type == "Genel":
            lines.append("Genel log bulunamadı.")

    return "\n".join(lines) if lines else "Henüz log kaydı yok. Bir işlem yapın (model yükleme, sohbet, arama)."


def clear_logs_action(table_choice):
    """Clear logs."""
    orch = _get_orch()
    table_map = {
        "Tümü": "all", "Arama": "search", "DSPy": "dspy",
        "LLM": "llm", "Hata": "errors", "Genel": "general",
    }
    success = orch.clear_logs(table_map.get(table_choice, "all"))
    return "✅ Loglar temizlendi." if success else "❌ Log temizleme başarısız."


# ═══════════════════════════════════════════════════════════════════
# CSS & THEME
# ═══════════════════════════════════════════════════════════════════

CSS = """
/* Minimal Header */
#header-compact {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    padding: 10px 16px;
    border-bottom: 1px solid #334155;
    margin-bottom: 0;
}
#header-compact h2 {
    color: #e2e8f0;
    font-size: 1.3em;
    margin: 0;
    font-weight: 600;
}

/* Sidebar styling */
.sidebar-panel {
    background: #f8fafc;
    border-right: 1px solid #e2e8f0;
    padding: 12px;
}

/* Footer status bar */
.footer-status {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    z-index: 100;
}

/* Chatbot height */
#chatbot-workspace {
    height: 550px !important;
    max-height: 65vh;
}

/* Accordion improvements */
.accordions-wrapper .accordion {
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    margin-bottom: 8px;
}
"""

THEME = gr.themes.Base(primary_hue="slate", neutral_hue="gray")


# ═══════════════════════════════════════════════════════════════════
# MAIN UI BUILDER - WORKSPACE LAYOUT
# ═══════════════════════════════════════════════════════════════════

def build_workspace_ui():
    """Build modern workspace UI with sidebar + main canvas."""

    with gr.Blocks(title="OpenVINO LLM Studio - Workspace", css=CSS) as demo:

        # ═══════════════════════════════════════
        # 1. COMPACT HEADER (Minimal)
        # ═══════════════════════════════════════
        with gr.Row(elem_id="header-compact"):
            gr.HTML("<h2>⚡ OpenVINO LLM Studio</h2>")
            with gr.Column(scale=1, elem_classes=["mode-selector"]):
                mode_selector = gr.Radio(
                    label="Kullanıcı Modu",
                    choices=["🔰 Basit", "⚙️ Orta", "🔬 Expert"],
                    value="⚙️ Orta",
                    show_label=False,
                    container=False,
                )

        # ═══════════════════════════════════════
        # 2. MAIN WORKSPACE (Sidebar + Canvas)
        # ═══════════════════════════════════════
        with gr.Row(equal_height=False):

            # ───────────────────────────────────
            # SOL PANEL: Sidebar (Ayarlar)
            # ───────────────────────────────────
            with gr.Column(scale=1, min_width=320, elem_classes=["sidebar-panel"]):

                # Model Selection Accordion
                with gr.Accordion("🤖 Model Seçimi", open=True, elem_classes=["accordions-wrapper"]):
                    backend_radio = gr.Radio(
                        label="Backend",
                        choices=["openvino", "ollama", "ipex"],
                        value="openvino",
                    )

                    with gr.Row():
                        model_dropdown = gr.Dropdown(
                            label="Model",
                            choices=[],
                            interactive=True,
                            allow_custom_value=True,
                        )
                        refresh_btn = gr.Button("🔄", size="sm", scale=0)

                    device_radio = gr.Radio(
                        label="Cihaz (GPU sadece Expert modda aktif)",
                        choices=["CPU", "GPU", "AUTO"],
                        value="CPU",
                        info="GPU: Intel Arc iGPU | AUTO: OpenVINO karar verir",
                    )

                    load_btn = gr.Button("⚡ Modeli Yükle", variant="primary", size="md")
                    load_status = gr.Textbox(
                        label="Model Durumu", 
                        value="Model yüklenmedi. Lütfen model seçin veya indirin.",
                        interactive=False, 
                        max_lines=2,
                        visible=True,
                    )

                # GPU Settings Accordion (Expert only)
                with gr.Accordion("⚙️ GPU/CPU Ayarları", open=False, visible=False, elem_id="gpu-settings") as gpu_accordion:
                    gr.Markdown("**Intel Arc iGPU** optimizasyonları")

                    gpu_max_alloc = gr.Slider(
                        label="GPU Max Alloc %",
                        minimum=40, maximum=95, value=75, step=5,
                        info="Paylaşımlı RAM limiti (75% önerilen)"
                    )

                    kv_cache_prec = gr.Radio(
                        label="KV Cache Precision",
                        choices=["f16", "u8"],
                        value="u8",
                        info="u8 = ~2x daha az VRAM"
                    )

                    num_streams = gr.Slider(
                        label="Num Streams",
                        minimum=1, maximum=4, value=1, step=1,
                    )

                    perf_hint = gr.Radio(
                        label="Performance Hint",
                        choices=["LATENCY", "THROUGHPUT"],
                        value="LATENCY",
                    )

                    cache_dir = gr.Textbox(
                        label="Cache Dizini",
                        value=str(CACHE_DIR),
                        info="Compiled kernel cache dizini",
                        visible=True,
                    )

                # LLM Parameters Accordion
                with gr.Accordion("📊 LLM Parametreleri", open=True, elem_classes=["accordions-wrapper"]):
                    preset_selector = gr.Radio(
                        label="Preset",
                        choices=["🚀 Hızlı", "⚖️ Dengeli", "🎨 Kaliteli"],
                        value="⚖️ Dengeli",
                    )

                    temperature = gr.Slider(
                        label="Temperature",
                        minimum=0.0, maximum=2.0, value=0.7, step=0.05,
                    )

                    max_tokens = gr.Slider(
                        label="Max Tokens",
                        minimum=64, maximum=4096, value=512, step=64,
                    )

                    top_p = gr.Slider(
                        label="Top-P",
                        minimum=0.1, maximum=1.0, value=0.9, step=0.05,
                    )

                    top_k = gr.Slider(
                        label="Top-K",
                        minimum=1, maximum=200, value=50, step=1,
                    )

                    repetition_penalty = gr.Slider(
                        label="Repetition Penalty",
                        minimum=1.0, maximum=2.0, value=1.1, step=0.05,
                    )

                # Search & DSPy Accordion
                with gr.Accordion("🔍 Arama & DSPy", open=True, elem_classes=["accordions-wrapper"]):
                    enable_search = gr.Checkbox(label="🌐 Web Araması", value=True)
                    enable_dspy = gr.Checkbox(label="🧠 DSPy", value=True)

                    num_search_results = gr.Slider(
                        label="Sonuç Sayısı",
                        minimum=1, maximum=10, value=5, step=1,
                    )

                # System Prompt (visible in medium/expert mode)
                with gr.Accordion("📝 Sistem Prompt", open=False, elem_classes=["accordions-wrapper"]) as system_prompt_accordion:
                    system_prompt = gr.Textbox(
                        label="Sistem Prompt (Opsiyonel)",
                        placeholder="Sen bir yardımcı asistansın...",
                        lines=3,
                        info="Modelin kişiliğini ve davranışını belirler",
                    )

            # ───────────────────────────────────
            # SAĞ PANEL: Ana Sohbet (Canvas)
            # ───────────────────────────────────
            with gr.Column(scale=3, min_width=500):

                # Chatbot (Merkez)
                chatbot = gr.Chatbot(
                    label="Sohbet",
                    height=550,
                    elem_id="chatbot-workspace",
                    avatar_images=(None, "https://cdn-icons-png.flaticon.com/512/4712/4712109.png"),
                )

                # DSPy/Arama Detayları (Genişletilebilir)
                with gr.Accordion("📋 Arama & DSPy Detayları", open=False, elem_classes=["accordions-wrapper"]):
                    pipeline_output = gr.Textbox(
                        label="Detaylı Çıktı",
                        lines=6,
                        interactive=False,
                        max_lines=10,
                    )

                # Prompt Input
                with gr.Row():
                    prompt_input = gr.Textbox(
                        label="Prompt",
                        placeholder="Sorunuzu buraya yazın...",
                        lines=3,
                        scale=4,
                        show_label=False,
                    )

                    with gr.Column(scale=1, min_width=120):
                        submit_btn = gr.Button("📤 Gönder", variant="primary", size="lg")
                        clear_btn = gr.Button("🗑️ Temizle", size="sm")

                pipeline_status = gr.Textbox(
                    label="Durum",
                    value="Hazır.",
                    interactive=False,
                    max_lines=1,
                    visible=False,
                )

        # ═══════════════════════════════════════
        # 3. FOOTER (Status Bar)
        # ═══════════════════════════════════════
        status_html = gr.HTML(
            value=format_footer_html(get_system_status()),
            elem_classes=["footer-status"],
        )

        # ═══════════════════════════════════════
        # 4. MODAL: Loglar (Ayrı pencere)
        # ═══════════════════════════════════════
        with gr.Accordion("📋 Loglar", open=False, elem_id="logs-modal"):
            with gr.Row():
                log_type = gr.Dropdown(
                    label="Log Türü",
                    choices=["Özet", "Arama", "DSPy", "LLM", "Hata", "Genel", "Ham JSON"],
                    value="Özet",
                )
                refresh_logs_btn = gr.Button("🔄 Yenile", size="sm")
                clear_logs_btn = gr.Button("🗑️ Logları Temizle", size="sm", variant="stop")

            log_display = gr.Textbox(label="Log İçeriği", lines=15, interactive=False)
            clear_logs_status = gr.Textbox(label="Temizleme Durumu", interactive=False, max_lines=1, visible=False)

            def clear_logs_wrapper():
                result = clear_logs_action("all")
                return result

            clear_logs_btn.click(
                fn=clear_logs_wrapper,
                outputs=[clear_logs_status],
            )

        # ═══════════════════════════════════════
        # EVENT HANDLERS
        # ═══════════════════════════════════════

        # Mode selector visibility
        def update_mode_visibility(mode):
            is_expert = mode == "🔬 Expert"
            is_basic = mode == "🔰 Basit"
            return [
                gr.update(visible=is_expert),  # GPU settings
                gr.update(visible=not is_basic),  # LLM params labels
                gr.update(visible=not is_basic),  # System prompt accordion
            ]

        mode_selector.change(
            fn=update_mode_visibility,
            inputs=[mode_selector],
            outputs=[gpu_accordion, temperature, system_prompt_accordion],
        )

        # Backend change
        def on_backend_change(b):
            _get_orch().set_backend(b)
            models, status = refresh_models()
            return models

        backend_radio.change(
            fn=on_backend_change,
            inputs=[backend_radio],
            outputs=[model_dropdown],
        )

        # Refresh models
        refresh_btn.click(
            fn=refresh_models,
            outputs=[model_dropdown, load_status],
        )

        # Load model
        load_btn.click(
            fn=load_model_action,
            inputs=[model_dropdown, device_radio, gpu_max_alloc, kv_cache_prec, num_streams, perf_hint, cache_dir],
            outputs=[load_status, status_html],
        )

        # Preset selector
        def apply_preset(preset):
            presets = {
                "🚀 Hızlı": (0.5, 256, 0.8, 40, 1.05),
                "⚖️ Dengeli": (0.7, 512, 0.9, 50, 1.1),
                "🎨 Kaliteli": (0.9, 1024, 0.95, 100, 1.15),
            }
            if preset in presets:
                t, m, p, k, r = presets[preset]
                return [gr.update(value=v) for v in [t, m, p, k, r]]
            return [gr.update() for _ in range(5)]

        preset_selector.change(
            fn=apply_preset,
            inputs=[preset_selector],
            outputs=[temperature, max_tokens, top_p, top_k, repetition_penalty],
        )

        # Submit
        submit_btn.click(
            fn=run_inference,
            inputs=[prompt_input, enable_search, enable_dspy, num_search_results,
                    temperature, max_tokens, top_p, top_k, repetition_penalty,
                    system_prompt, chatbot],
            outputs=[chatbot, pipeline_output, status_html],
        )

        # Clear chat
        clear_btn.click(
            fn=lambda: ([], ""),
            outputs=[chatbot, pipeline_output],
        )

        # Refresh logs
        refresh_logs_btn.click(
            fn=lambda lt: get_logs_display(lt, True),
            inputs=[log_type],
            outputs=[log_display],
        )

        # Initial load
        demo.load(
            fn=refresh_models,
            outputs=[model_dropdown, load_status],
        )

    return demo


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    demo = build_workspace_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7862,
        show_error=True,
        theme=THEME,
    )
