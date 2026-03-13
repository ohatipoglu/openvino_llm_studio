"""
ui/app.py
Gradio tabanlı ana kullanıcı arayüzü.
Tüm parametreler kullanıcı tarafından kontrol edilebilir.

Değişiklikler (IPEX-LLM EOL → llama-cpp-python):
  - "IPEX Worker Başlat/Durdur" → "llama-cpp Hazırlık Kontrolü"
  - Model Galerisi IPEX sekmesi → GGUF sekmesi
  - GGUF dosyası indirme + yerel GGUF tarama desteği
  - download_ipex_model → download_gguf_model
"""

import sys
import os
import json
import logging
import threading
import concurrent.futures
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import gradio as gr
from core.orchestrator import Orchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(ROOT / "logs" / "studio.log", encoding="utf-8"),
    ]
)
logger = logging.getLogger(__name__)

_orch_lock = threading.Lock()
orch = Orchestrator()


# ===================== HELPER FUNCTIONS =====================

def refresh_models():
    choices = orch.get_model_choices()
    if choices:
        return gr.update(choices=choices, value=choices[0]), f"✅ {len(choices)} model bulundu."
    return gr.update(choices=[], value=None), "⚠️ Model bulunamadı."


def load_model_action(model_choice, device,
                       gpu_max_alloc, kv_cache_prec, num_streams, perf_hint, cache_dir):
    if not model_choice:
        return "❌ Model seçilmedi.", gr.update(interactive=False)

    # GPU/AUTO için ov_config oluştur
    ov_config = None
    if device.upper() in ("GPU", "AUTO"):
        ov_config = {
            "GPU_MAX_ALLOC_PERCENT": str(int(gpu_max_alloc)),
            "KV_CACHE_PRECISION":    kv_cache_prec,
            "NUM_STREAMS":           str(int(num_streams)),
            "PERFORMANCE_HINT":      perf_hint,
        }
        if cache_dir.strip():
            ov_config["CACHE_DIR"] = cache_dir.strip()
    elif device.upper() == "CPU":
        import os
        cpu_count = os.cpu_count() or 8
        ov_config = {
            "INFERENCE_NUM_THREADS": str(max(4, cpu_count - 2)),
            "NUM_STREAMS":           str(int(num_streams)),
            "PERFORMANCE_HINT":      perf_hint,
            "CPU_BIND_THREAD":       "YES",
        }
        if cache_dir.strip():
            ov_config["CACHE_DIR"] = cache_dir.strip()

    success, msg = orch.load_model(model_choice, device, ov_config=ov_config)
    return msg, gr.update(interactive=success)


def run_inference(
    prompt, system_prompt,
    enable_search, enable_dspy,
    num_search_results, search_region,
    temperature, max_tokens, top_p, top_k,
    repetition_penalty,
    history
):
    if not prompt.strip():
        yield history, "⚠️ Prompt boş olamaz."
        return

    if not orch.is_model_loaded:
        yield history, "❌ Model yüklü değil. Lütfen sol panelden bir model seçin ve yükleyin."
        return

    params = {
        "temperature":      temperature,
        "max_tokens":       int(max_tokens),
        "top_p":            top_p,
        "top_k":            int(top_k),
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
        yield history, accumulated

    history = list(history)
    history.append({"role": "user",      "content": prompt})
    history.append({"role": "assistant", "content": accumulated})
    yield history, accumulated


def get_logs_display(log_type, session_only):
    logs = orch.get_logs(session_only=session_only)

    if log_type == "Ham JSON":
        return json.dumps(logs, ensure_ascii=False, indent=2, default=str)

    lines = []

    if log_type in ("Özet", "Arama"):
        searches = logs.get("search", [])
        if searches:
            lines += ["━" * 60, f"🔍 ARAMA LOGLARI ({len(searches)} kayıt)", "━" * 60]
            for r in searches:
                lines += [
                    f"\n📅 {r.get('timestamp','')}",
                    f"   Orijinal Prompt : {r.get('original_prompt','')}",
                    f"   Arama Sorgusu   : {r.get('search_query','')}",
                    f"   Sonuç Sayısı    : {r.get('num_results',0)}",
                    f"   Süre            : {r.get('duration_ms',0):.0f} ms",
                ]
                ranked = r.get("ranked_results")
                if ranked:
                    try:
                        results = json.loads(ranked) if isinstance(ranked, str) else ranked
                        lines.append("   Sıralı Sonuçlar :")
                        for res in results[:5]:
                            lines.append(f"      [{res.get('rank','-')}] {res.get('title','')[:70]}")
                            lines.append(f"          {res.get('url','')[:80]}")
                    except Exception:
                        pass
                lines.append("")

    if log_type in ("Özet", "DSPy"):
        dspy_logs = logs.get("dspy", [])
        if dspy_logs:
            lines += ["━" * 60, f"🧠 DSPy LOGLARI ({len(dspy_logs)} kayıt)", "━" * 60]
            for r in dspy_logs:
                lines += [
                    f"\n📅 {r.get('timestamp','')}",
                    f"   Seçilen Mod      : {r.get('detected_mode','')}",
                    f"   Mod Gerekçesi    : {r.get('mode_reason','')}",
                    f"   Süre             : {r.get('duration_ms',0):.0f} ms",
                    f"   Zenginleştirilmiş Prompt (ilk 300):",
                    f"      {str(r.get('enriched_prompt',''))[:300].replace(chr(10),' | ')}",
                ]
                lines.append("")

    if log_type in ("Özet", "LLM"):
        llm_logs = logs.get("llm", [])
        if llm_logs:
            lines += ["━" * 60, f"⚡ LLM LOGLARI ({len(llm_logs)} kayıt)", "━" * 60]
            for r in llm_logs:
                lines += [
                    f"\n📅 {r.get('timestamp','')}",
                    f"   Model            : {r.get('model_name','')}",
                    f"   Input/Output     : {r.get('input_tokens',0)} / {r.get('output_tokens',0)} token",
                    f"   Tok/sn | Süre    : {r.get('tokens_per_second',0):.2f} | {r.get('duration_ms',0):.0f} ms",
                    f"   Yanıt (ilk 400)  :",
                    f"      {str(r.get('response',''))[:400].replace(chr(10),' | ')}",
                ]
                lines.append("")

    if log_type in ("Özet", "Hata"):
        errors = logs.get("errors", [])
        if errors:
            lines += ["━" * 60, f"❌ HATA LOGLARI ({len(errors)} kayıt)", "━" * 60]
            for r in errors:
                lines += [
                    f"\n📅 {r.get('timestamp','')}",
                    f"   Modül      : {r.get('module','')}",
                    f"   Hata Tipi  : {r.get('error_type','')}",
                    f"   Mesaj      : {r.get('error_message','')}",
                ]
                for stline in str(r.get("stack_trace","")).split("\n")[-5:]:
                    lines.append(f"      {stline}")
                lines.append("")

    if log_type in ("Özet", "Genel"):
        general = logs.get("general", [])
        if general:
            lines += ["━" * 60, f"📝 GENEL LOGLAR ({len(general)} kayıt)", "━" * 60]
            for r in general:
                icon = {"INFO": "ℹ️", "WARNING": "⚠️", "ERROR": "❌", "DEBUG": "🔧"}.get(
                    r.get("level",""), "•"
                )
                lines.append(f"{icon} {r.get('timestamp','')}  [{r.get('module','')}]  {r.get('message','')}")

    return "\n".join(lines) if lines else "Henüz bu türde log kaydı yok."


def clear_logs_action(table_choice):
    table_map = {
        "Tümü": "all", "Arama": "search", "DSPy": "dspy",
        "LLM": "llm", "Hata": "errors", "Genel": "general",
    }
    success = orch.clear_logs(table_map.get(table_choice, "all"))
    return "✅ Loglar temizlendi." if success else "❌ Log temizleme başarısız."


def get_stats_display():
    stats          = orch.get_stats()
    backend_status = stats.get("backend_status", {})
    lines = [
        f"🔖 Session: {stats.get('session_id','?')}",
        f"🔧 Aktif Backend: {stats.get('active_backend','?').upper()}",
        f"🤖 Yüklü Model: {stats.get('model_loaded','Yok')}",
        "",
        "🔌 Backend Durumları:",
        f"  OpenVINO   : {backend_status.get('openvino','?')}",
        f"  Ollama     : {backend_status.get('ollama','?')}",
        f"  llama-cpp  : {backend_status.get('ipex','?')}",
        "",
        "💾 Bellek:",
        f"  Toplam    : {stats.get('total_gb','?')} GB",
        f"  Kullanılan: {stats.get('used_gb','?')} GB",
        f"  Müsait    : {stats.get('available_gb','?')} GB  ({stats.get('percent','?')}%)",
        "",
        "📊 Veritabanı:",
        f"  Arama sayısı : {stats.get('total_searches',0)}",
        f"  LLM çağrısı  : {stats.get('total_llm_calls',0)}",
        f"  DSPy çağrısı : {stats.get('total_dspy_calls',0)}",
        f"  Hata sayısı  : {stats.get('total_errors',0)}",
        f"  DB boyutu    : {stats.get('db_size_mb',0)} MB",
    ]
    return "\n".join(lines)


def new_session_action():
    orch.new_session()
    return [], f"🔄 Yeni session: {orch.session_id}"


# ===================== KATALOG YARDIMCI FONKSİYONLARI =====================

def _entries_to_table_ov(entries):
    rows, choices = [], []
    for e in entries:
        lock = "🔒 " if e.gated else ""
        rows.append([lock + e.name, e.model_id, e.size_str, e.context,
                     e.downloads, ", ".join(e.tags[:4])])
        choices.append(f"{e.model_id}  ({e.size_str})")
    return rows, gr.update(choices=choices, value=choices[0] if choices else None)


def _entries_to_table_gguf(entries):
    rows, choices = [], []
    for e in entries:
        rows.append([e.name, e.model_id, e.size_str, e.context,
                     e.downloads, ", ".join(e.tags[:4])])
        choices.append(f"{e.model_id}  ({e.size_str})")
    return rows, gr.update(choices=choices, value=choices[0] if choices else None)


def _entries_to_table_ollama(entries):
    rows, choices = [], []
    for e in entries:
        rows.append([e.name, e.model_id, e.size_str, e.context,
                     ", ".join(e.tags[:4])])
        choices.append(f"{e.model_id}  ({e.size_str})")
    return rows, gr.update(choices=choices, value=choices[0] if choices else None)


# ===================== KATALOG YENİLEME (ZAMAN AŞIMLI) =====================

def refresh_ov_catalog(search, force=False):
    yield [], gr.update(choices=[], value=None), "⏳ HuggingFace'den OpenVINO modelleri çekiliyor..."
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(orch.get_openvino_catalog, search.strip(), force)
        try:
            entries = future.result(timeout=30)
        except concurrent.futures.TimeoutError:
            yield [], gr.update(choices=[], value=None), "❌ Zaman aşımı: API çok yavaş."
            return
        except Exception as e:
            yield [], gr.update(choices=[], value=None), f"❌ Hata: {e}"
            return
    rows, dropdown = _entries_to_table_ov(entries)
    yield rows, dropdown, f"✅ {len(entries)} OpenVINO modeli listelendi."


def refresh_ollama_catalog(search, force=False):
    yield [], gr.update(choices=[], value=None), "⏳ Ollama kataloğu yükleniyor..."
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(orch.get_ollama_catalog, search.strip(), force)
        try:
            entries = future.result(timeout=30)
        except concurrent.futures.TimeoutError:
            yield [], gr.update(choices=[], value=None), "❌ Zaman aşımı."
            return
        except Exception as e:
            yield [], gr.update(choices=[], value=None), f"❌ Hata: {e}"
            return
    rows, dropdown = _entries_to_table_ollama(entries)
    yield rows, dropdown, f"✅ {len(entries)} Ollama modeli listelendi."


def refresh_gguf_catalog(search, force=False):
    yield [], gr.update(choices=[], value=None), "⏳ GGUF model kataloğu yükleniyor..."
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(orch.get_ipex_catalog, search.strip(), force)
        try:
            entries = future.result(timeout=30)
        except concurrent.futures.TimeoutError:
            yield [], gr.update(choices=[], value=None), "❌ Zaman aşımı."
            return
        except Exception as e:
            yield [], gr.update(choices=[], value=None), f"❌ Hata: {e}"
            return
    rows, dropdown = _entries_to_table_gguf(entries)
    yield rows, dropdown, f"✅ {len(entries)} GGUF modeli listelendi."


# ===================== DİĞER YARDIMCI FONKSİYONLAR =====================

def pull_ollama(model_id):
    if not model_id.strip():
        return "⚠️ Model ID boş olamaz."
    success, msg = orch.pull_ollama_model(model_id.strip())
    return f"✅ {msg}" if success else f"❌ {msg}"


def check_llamacpp():
    """llama-cpp-python kurulumunu kontrol et ve durum raporla."""
    ok, msg = orch.ipex.start_worker()
    status  = orch.get_backend_status()
    lines   = [f"{b.upper()}: {v}" for b, v in status.items()]
    return "\n".join(lines), f"{'✅' if ok else '❌'} {msg}"


def unload_llamacpp():
    """Yüklü GGUF modelini bellekten çıkar."""
    ok, msg = orch.ipex.stop_worker()
    status  = orch.get_backend_status()
    lines   = [f"{b.upper()}: {v}" for b, v in status.items()]
    return "\n".join(lines), f"{'✅' if ok else '❌'} {msg}"


def on_backend_change(backend):
    orch.set_backend(backend)
    status  = orch.get_backend_status()
    lines   = [f"{b.upper()}: {v}" for b, v in status.items()]
    choices = orch.get_model_choices()
    val     = choices[0] if choices else None
    return "\n".join(lines), gr.update(choices=choices, value=val)


def dl_ov_catalog(selected, dest_dir):
    if not selected:
        return "⚠️ Model seçilmedi."
    model_id = selected.split("  (")[0].strip()
    yield f"⏳ '{model_id}' indiriliyor → {dest_dir} ..."
    success, msg = orch.download_openvino_model(model_id, dest_dir)
    yield msg


def load_ov_catalog(selected, device):
    if not selected:
        return "⚠️ Model seçilmedi."
    model_id   = selected.split("  (")[0].strip()
    local_name = model_id.replace("/", "--")
    dest       = os.path.join(r"C:\OpenVINO_LLM", local_name)
    orch.set_backend("openvino")
    success, msg = orch.load_model(dest, device)
    return f"✅ {msg}" if success else f"❌ {msg}"


def pull_ollama_catalog(selected):
    if not selected:
        return "⚠️ Model seçilmedi."
    model_id = selected.split("  (")[0].strip()
    yield f"⏳ '{model_id}' Ollama'ya indiriliyor..."
    success, msg = orch.pull_ollama_model(model_id)
    yield f"✅ {msg}" if success else f"❌ {msg}"


def load_ollama_catalog(selected):
    if not selected:
        return "⚠️ Model seçilmedi."
    model_id = selected.split("  (")[0].strip()
    orch.set_backend("ollama")
    success, msg = orch.load_model(model_id)
    return f"✅ {msg}" if success else f"❌ {msg}"


def dl_gguf_catalog(selected, gguf_quant, dest_dir):
    """
    HF'den GGUF dosyası indir.
    selected:    "org/repo-GGUF  (~4.3 GB)"
    gguf_quant:  "Q4_K_M", "Q4_K_S", "Q5_K_M", "Q8_0", "otomatik"
    dest_dir:    hedef dizin
    """
    if not selected:
        return "⚠️ Model seçilmedi."
    model_id = selected.split("  (")[0].strip()

    # Quant seçimi
    filename = ""
    if gguf_quant and gguf_quant.lower() != "otomatik":
        # Tam dosya adını bilmiyoruz, arama yapacak
        filename = ""   # download_gguf_model içinde seçilecek

    os.makedirs(dest_dir, exist_ok=True)
    yield f"⏳ '{model_id}' GGUF indiriliyor → {dest_dir}\nSeçilen quant: {gguf_quant} ..."
    success, msg = orch.download_gguf_model(model_id, filename=filename, dest_dir=dest_dir)
    yield msg


def load_gguf_catalog(selected, device, gguf_n_ctx, gguf_force_cpu):
    """Yerel .gguf dosyasını seçip yükle."""
    if not selected:
        return "⚠️ Model seçilmedi."
    model_id = selected.split("  (")[0].strip()

    # Yerel GGUF taramasından tam yolu bul
    local_models = orch.ipex.scan_local_gguf()
    match = next((m for m in local_models if m.model_id == model_id
                  or Path(m.model_id).stem == model_id), None)

    if match:
        path = match.model_id
    elif Path(model_id).exists():
        path = model_id
    else:
        return (
            f"⚠️ '{model_id}' yerel olarak bulunamadı.\n"
            "Önce 'HF'den İndir' butonunu kullanın."
        )

    orch.set_backend("ipex")

    # Cihaz: force_cpu işaretliyse CPU, değilse UI seçimine göre
    if gguf_force_cpu:
        ipex_device = "cpu"
    else:
        ipex_device = "xpu" if device.upper() in ("GPU", "AUTO") else "cpu"

    n_ctx = max(512, int(gguf_n_ctx))

    success, msg = orch.ipex.load(path, device=ipex_device, n_ctx=n_ctx,
                                     session_id=orch.session_id)
    return msg


def _initial_load():
    ov_e = orch.get_openvino_catalog()
    ov_r, ov_d = _entries_to_table_ov(ov_e)

    ol_e = orch.get_ollama_catalog()
    ol_r, ol_d = _entries_to_table_ollama(ol_e)

    ip_e = orch.get_ipex_catalog()
    ip_r, ip_d = _entries_to_table_gguf(ip_e)

    return (ov_r, ov_d, f"✅ {len(ov_e)} model",
            ol_r, ol_d, f"✅ {len(ol_e)} model",
            ip_r, ip_d, f"✅ {len(ip_e)} model")


# ===================== GRADIO UI =====================

CSS = """
#header { background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #1e3a5f 100%);
          padding: 20px; border-radius: 12px; margin-bottom: 16px; }
#header h1 { color: #e0e7ff; font-size: 1.8em; margin: 0; }
#header p  { color: #a5b4fc; margin: 4px 0 0 0; font-size: 0.9em; }
.model-status { font-family: monospace; font-size: 0.85em; }
#response-box textarea { font-family: 'JetBrains Mono', monospace; font-size: 0.9em; }
.tab-nav button { font-weight: 600; }
"""

THEME = gr.themes.Base(primary_hue="indigo", neutral_hue="slate")


def build_ui():
    with gr.Blocks(title="OpenVINO LLM Studio") as demo:

        gr.HTML("""
        <div id="header">
          <h1>⚡ OpenVINO LLM Studio</h1>
          <p>Intel OpenVINO · Ollama · llama-cpp-python (GGUF/SYCL) · DuckDuckGo · DSPy · SQLite</p>
        </div>
        """)

        with gr.Tabs():

            # ═══════════════════════════════════════
            # TAB 1: ANA CHAT
            # ═══════════════════════════════════════
            with gr.TabItem("💬 Chat"):
                with gr.Row():
                    # Sol panel
                    with gr.Column(scale=1, min_width=300):
                        gr.Markdown("### 🔧 Backend Seçimi")

                        backend_radio = gr.Radio(
                            label="Inference Backend",
                            choices=["openvino", "ollama", "ipex"],
                            value="openvino",
                            info=(
                                "OpenVINO: .xml/.bin modeller  |  "
                                "Ollama: yerel Ollama sunucu  |  "
                                "llama-cpp: GGUF modeller (Intel Arc SYCL veya CPU)"
                            )
                        )
                        backend_status_box = gr.Textbox(
                            label="Backend Durumu", interactive=False,
                            elem_classes=["model-status"], lines=3
                        )
                        with gr.Row():
                            lcpp_check_btn = gr.Button("🔍 llama-cpp Kontrol", scale=1, size="sm")
                            lcpp_unload_btn = gr.Button("⏹️ Modeli Boşalt",    scale=0, size="sm", variant="stop")

                        gr.Markdown("---")
                        gr.Markdown("### 🤖 Model Seçimi")

                        with gr.Row():
                            model_dropdown = gr.Dropdown(
                                label="Model", choices=[], interactive=True,
                                elem_id="model-drop"
                            )
                            refresh_btn = gr.Button("🔄", scale=0, min_width=50)

                        device_radio = gr.Radio(
                            label="Cihaz (OpenVINO / llama-cpp SYCL)",
                            choices=["CPU", "GPU", "AUTO"], value="CPU"
                        )
                        with gr.Accordion("⚙️ Gelişmiş GPU/CPU Ayarları", open=False):
                            gr.Markdown(
                                "**Intel Arc iGPU** kullanıyorsanız bu ayarlar "
                                "`dxgmms2.sys` crash'ini önler.\n\n"
                                "**GPU_MAX_ALLOC_PERCENT** → paylaşımlı RAM'den "
                                "ne kadar kullanılacağını sınırlar.  \n"
                                "32 GB × 75% = ~24 GB max — sistem kararlı kalır."
                            )
                            gpu_max_alloc = gr.Slider(
                                label="GPU Max Alloc % — iGPU paylaşımlı RAM limiti",
                                minimum=40, maximum=95, value=75, step=5,
                                info="↓ Düşür = dxgmms2.sys crash riski azalır. 75% önerilen başlangıç."
                            )
                            kv_cache_prec = gr.Radio(
                                label="KV Cache Precision",
                                choices=["f16", "u8"], value="u8",
                                info="u8 = ~2x daha az VRAM. Büyük modellerde u8 kullan."
                            )
                            num_streams = gr.Slider(
                                label="Num Streams",
                                minimum=1, maximum=4, value=1, step=1,
                                info="Büyük modellerde 1 tut. Yüksek değer bellek artırır."
                            )
                            perf_hint = gr.Radio(
                                label="Performance Hint",
                                choices=["LATENCY", "THROUGHPUT"], value="LATENCY",
                                info="LATENCY = tek kullanıcı, düşük bellek overhead."
                            )
                            cache_dir = gr.Textbox(
                                label="OpenVINO Cache Dizini",
                                value=r"C:\OpenVINO_LLM\.cache",
                                info="Compiled kernel cache. Boş bırakırsan cache kapalı."
                            )
                        load_btn    = gr.Button("⚡ Modeli Yükle", variant="primary")
                        load_status = gr.Textbox(
                            label="Yükleme Durumu", value="Model yüklenmedi.",
                            interactive=False, elem_classes=["model-status"]
                        )

                        with gr.Accordion("🦙 Ollama — Model İndir", open=False):
                            gr.Markdown("Örnek: `qwen2.5:7b`, `qwen3:8b`, `mistral:7b`")
                            ollama_pull_input  = gr.Textbox(label="Model ID", placeholder="qwen2.5:7b")
                            ollama_pull_btn    = gr.Button("⬇️ İndir", variant="secondary")
                            ollama_pull_status = gr.Textbox(label="İndirme Durumu", interactive=False, max_lines=2)

                        gr.Markdown("---")
                        gr.Markdown("### 🔍 Arama & DSPy")

                        enable_search = gr.Checkbox(label="🌐 Web Araması",         value=True)
                        enable_dspy   = gr.Checkbox(label="🧠 DSPy Zenginleştirme", value=True)
                        num_results   = gr.Slider(label="Arama Sonuç Sayısı",
                                                  minimum=1, maximum=10, value=5, step=1)
                        search_region = gr.Dropdown(
                            label="Arama Bölgesi",
                            choices=["tr-tr", "en-us", "de-de", "fr-fr", "wt-wt"],
                            value="tr-tr"
                        )

                        gr.Markdown("---")
                        gr.Markdown("### ⚙️ LLM Parametreleri")

                        temperature  = gr.Slider(label="Temperature",    minimum=0.0, maximum=2.0, value=0.7, step=0.05)
                        max_tokens   = gr.Slider(label="Max New Tokens",  minimum=64, maximum=4096, value=512, step=64)
                        top_p        = gr.Slider(label="Top-P",           minimum=0.1, maximum=1.0, value=0.9, step=0.05)
                        top_k        = gr.Slider(label="Top-K",           minimum=1, maximum=200, value=50, step=1)
                        rep_penalty  = gr.Slider(label="Repetition Penalty", minimum=1.0, maximum=2.0, value=1.1, step=0.05)

                        gr.Markdown("---")
                        gr.Markdown("### 📝 Sistem Prompt")
                        system_prompt_box = gr.Textbox(
                            label="System Prompt (opsiyonel)",
                            placeholder="Sen bir yardımcı asistansın...", lines=3
                        )

                    # Sağ panel
                    with gr.Column(scale=2):
                        gr.Markdown("### 💬 Konuşma")
                        chatbot = gr.Chatbot(label="Sohbet Geçmişi", height=400)

                        pipeline_output = gr.Textbox(
                            label="Pipeline Çıktısı (Arama + DSPy + LLM)",
                            lines=12, interactive=False, elem_id="response-box"
                        )

                        with gr.Row():
                            prompt_input = gr.Textbox(
                                label="Prompt", placeholder="Sorunuzu buraya yazın...",
                                lines=3, scale=4
                            )
                            with gr.Column(scale=1, min_width=100):
                                submit_btn    = gr.Button("📤 Gönder", variant="primary")
                                clear_chat_btn = gr.Button("🗑️ Temizle")
                                new_session_btn = gr.Button("🔄 Yeni Session")

                        pipeline_status = gr.Textbox(
                            label="Durum", value="Hazır.", interactive=False, max_lines=1
                        )

            # ═══════════════════════════════════════
            # TAB 2: LOGLAR
            # ═══════════════════════════════════════
            with gr.TabItem("📋 Loglar"):
                gr.Markdown("### 📋 İşlem Logları")
                with gr.Row():
                    log_type      = gr.Dropdown(
                        label="Log Türü",
                        choices=["Özet", "Arama", "DSPy", "LLM", "Hata", "Genel", "Ham JSON"],
                        value="Özet"
                    )
                    session_only_cb  = gr.Checkbox(label="Sadece bu session", value=True)
                    refresh_logs_btn = gr.Button("🔄 Yenile", variant="secondary")

                log_display = gr.Textbox(label="Log İçeriği", lines=30, interactive=False)

                gr.Markdown("### 🗑️ Log Temizleme")
                with gr.Row():
                    clear_table    = gr.Dropdown(
                        label="Silinecek Tablo",
                        choices=["Tümü", "Arama", "DSPy", "LLM", "Hata", "Genel"],
                        value="Tümü"
                    )
                    clear_logs_btn = gr.Button("🗑️ Seçili Logları Sil", variant="stop")
                clear_status = gr.Textbox(label="Silme Durumu", interactive=False, max_lines=1)

            # ═══════════════════════════════════════
            # TAB 3: MODEL GALERİSİ
            # ═══════════════════════════════════════
            with gr.TabItem("🛍️ Model Galerisi"):
                gr.Markdown("### 🛍️ Model Galerisi — Canlı HuggingFace Kataloğu")
                gr.Markdown(
                    "Modeller **HuggingFace Hub'dan canlı** çekilir (30 dk önbelleklenir). "
                    "İnternet bağlantısı yoksa yerleşik fallback listesi kullanılır."
                )

                with gr.Tabs():

                    # ── OpenVINO ──────────────────────────────────
                    with gr.TabItem("⚡ OpenVINO Modelleri"):
                        gr.Markdown(
                            "HF'de `library=openvino` etiketli modeller. "
                            "İndirilen model `C:\\OpenVINO_LLM\\` altına kopyalanır."
                        )
                        with gr.Row():
                            ov_search    = gr.Textbox(label="🔎 Ara", placeholder="qwen, mistral...", scale=3)
                            ov_refresh_btn = gr.Button("🔄 HF'den Yenile", scale=1)
                            ov_dest_dir  = gr.Textbox(label="İndirme Dizini", value=r"C:\OpenVINO_LLM", scale=2)

                        ov_catalog_table = gr.Dataframe(
                            headers=["Model Adı", "HF Model ID", "Boyut", "Context", "İndirme", "Etiketler"],
                            datatype=["str","str","str","str","number","str"],
                            interactive=False, wrap=True,
                            label="OpenVINO Model Kataloğu"
                        )
                        with gr.Row():
                            ov_catalog_select   = gr.Dropdown(label="Seçilen Model", choices=[], interactive=True, scale=3)
                            ov_catalog_dl_btn   = gr.Button("⬇️ HF'den İndir",  variant="primary",   scale=1)
                            ov_catalog_load_btn = gr.Button("⚡ Yükle (yerel)", variant="secondary", scale=1)
                        ov_catalog_status = gr.Textbox(label="Durum", interactive=False, lines=2)

                    # ── Ollama ────────────────────────────────────
                    with gr.TabItem("🦙 Ollama Modelleri"):
                        gr.Markdown(
                            "Standart Ollama veya **IPEX Ollama fork** ile iGPU hızlandırma. "
                            "✅ işareti yüklü modelleri gösterir."
                        )
                        with gr.Row():
                            ollama_search = gr.Textbox(label="🔎 Ara", placeholder="qwen, mistral...", scale=3)
                            ollama_refresh_catalog_btn = gr.Button("🔄 Yenile", scale=1)

                        ollama_catalog_table = gr.Dataframe(
                            headers=["Model Adı", "Ollama ID", "Boyut", "Context", "Etiketler"],
                            datatype=["str","str","str","str","str"],
                            interactive=False, wrap=True,
                            label="Ollama Model Kataloğu"
                        )
                        with gr.Row():
                            ollama_catalog_select   = gr.Dropdown(label="Seçilen Model", choices=[], interactive=True, scale=3)
                            ollama_catalog_pull_btn = gr.Button("⬇️ Ollama'ya İndir", variant="primary",   scale=1)
                            ollama_catalog_load_btn = gr.Button("⚡ Seç & Yükle",      variant="secondary", scale=1)
                        ollama_catalog_status = gr.Textbox(label="Durum", interactive=False, lines=2)

                    # ── GGUF / llama-cpp ──────────────────────────
                    with gr.TabItem("🔷 GGUF / llama-cpp Modelleri"):
                        gr.Markdown(
                            "**llama-cpp-python** ile çalışan GGUF modeller.\n\n"
                            "- **CPU**: `pip install llama-cpp-python`\n"
                            "- **Intel Arc XPU** (SYCL): "
                            "`set CMAKE_ARGS=-DGGML_SYCL=ON && pip install llama-cpp-python --no-binary llama-cpp-python`\n"
                            "- GGUF dosyaları `C:\\OpenVINO_LLM\\gguf\\` dizinine indirilir.\n"
                            "- ✅ işareti yerel olarak indirilen modelleri gösterir."
                        )
                        with gr.Row():
                            gguf_search      = gr.Textbox(label="🔎 Ara", placeholder="qwen, mistral, phi...", scale=3)
                            gguf_refresh_btn = gr.Button("🔄 HF'den Yenile", scale=1)
                            gguf_quant       = gr.Dropdown(
                                label="Quant Türü",
                                choices=["otomatik", "Q4_K_M", "Q4_K_S", "Q5_K_M", "Q8_0"],
                                value="otomatik", scale=1
                            )
                            gguf_dest_dir = gr.Textbox(
                                label="İndirme Dizini",
                                value=r"C:\OpenVINO_LLM\gguf", scale=2
                            )

                        gguf_catalog_table = gr.Dataframe(
                            headers=["Model Adı", "HF / GGUF Repo", "Boyut", "Context", "İndirme", "Etiketler"],
                            datatype=["str","str","str","str","number","str"],
                            interactive=False, wrap=True,
                            label="GGUF Model Kataloğu (HF Canlı)"
                        )
                        with gr.Row():
                            gguf_catalog_select   = gr.Dropdown(label="Seçilen Model", choices=[], interactive=True, scale=3)
                            gguf_catalog_dl_btn   = gr.Button("⬇️ HF'den İndir", variant="primary",   scale=1)
                            gguf_catalog_load_btn = gr.Button("⚡ Yükle (yerel)", variant="secondary", scale=1)

                        with gr.Accordion("⚙️ GGUF Yükleme Ayarları", open=True):
                            gguf_n_ctx = gr.Slider(
                                label="Context Uzunluğu (n_ctx)",
                                minimum=512, maximum=32768, value=4096, step=512,
                                info="Model 131k eğitildi ama RAM için 4096-8192 önerilen. Büyük ctx = daha fazla RAM."
                            )
                            gguf_force_cpu = gr.Checkbox(
                                label="⚠️ CPU'ya zorla (SYCL yoksa bunu işaretle)",
                                value=True,
                                info="SYCL derlemesi olmadan GPU seçersen model yüklenemez. "
                                     "CPU ile kesin çalışır."
                            )

                        gguf_catalog_status = gr.Textbox(label="Durum", interactive=False, lines=3)

            # ═══════════════════════════════════════
            # TAB 4: SİSTEM
            # ═══════════════════════════════════════
            with gr.TabItem("📊 Sistem"):
                gr.Markdown("### 📊 Sistem İstatistikleri")
                stats_display = gr.Textbox(
                    label="Durum", lines=16, interactive=False,
                    elem_classes=["model-status"]
                )
                refresh_stats_btn = gr.Button("🔄 Yenile", variant="secondary")

        # ===================== EVENT HANDLERS =====================

        # Katalog yenileme
        ov_search.submit(refresh_ov_catalog,     [ov_search],   [ov_catalog_table, ov_catalog_select, ov_catalog_status])
        ov_refresh_btn.click(refresh_ov_catalog, [ov_search],   [ov_catalog_table, ov_catalog_select, ov_catalog_status])

        ollama_search.submit(refresh_ollama_catalog,           [ollama_search], [ollama_catalog_table, ollama_catalog_select, ollama_catalog_status])
        ollama_refresh_catalog_btn.click(refresh_ollama_catalog, [ollama_search], [ollama_catalog_table, ollama_catalog_select, ollama_catalog_status])

        gguf_search.submit(refresh_gguf_catalog,     [gguf_search], [gguf_catalog_table, gguf_catalog_select, gguf_catalog_status])
        gguf_refresh_btn.click(refresh_gguf_catalog, [gguf_search], [gguf_catalog_table, gguf_catalog_select, gguf_catalog_status])

        # İndirme ve yükleme
        ov_catalog_dl_btn.click(dl_ov_catalog,     [ov_catalog_select, ov_dest_dir],       [ov_catalog_status])
        ov_catalog_load_btn.click(load_ov_catalog, [ov_catalog_select, device_radio],      [ov_catalog_status])

        ollama_catalog_pull_btn.click(pull_ollama_catalog, [ollama_catalog_select],         [ollama_catalog_status])
        ollama_catalog_load_btn.click(load_ollama_catalog, [ollama_catalog_select],         [ollama_catalog_status])

        gguf_catalog_dl_btn.click(dl_gguf_catalog,   [gguf_catalog_select, gguf_quant, gguf_dest_dir], [gguf_catalog_status])
        gguf_catalog_load_btn.click(
            load_gguf_catalog,
            inputs=[gguf_catalog_select, device_radio, gguf_n_ctx, gguf_force_cpu],
            outputs=[gguf_catalog_status]
        )

        # İlk yüklemede katalog doldur
        demo.load(
            _initial_load,
            outputs=[ov_catalog_table, ov_catalog_select, ov_catalog_status,
                     ollama_catalog_table, ollama_catalog_select, ollama_catalog_status,
                     gguf_catalog_table, gguf_catalog_select, gguf_catalog_status]
        )

        # Backend değişimi
        backend_radio.change(on_backend_change, [backend_radio], [backend_status_box, model_dropdown])

        # llama-cpp kontrol
        lcpp_check_btn.click(check_llamacpp,  outputs=[backend_status_box, load_status])
        lcpp_unload_btn.click(unload_llamacpp, outputs=[backend_status_box, load_status])

        # Ollama pull
        ollama_pull_btn.click(pull_ollama, [ollama_pull_input], [ollama_pull_status])

        # Model yenile ve yükle
        refresh_btn.click(refresh_models,   outputs=[model_dropdown, load_status])
        load_btn.click(
            load_model_action,
            inputs=[model_dropdown, device_radio,
                    gpu_max_alloc, kv_cache_prec, num_streams, perf_hint, cache_dir],
            outputs=[load_status, submit_btn]
        )

        # Chat
        submit_btn.click(
            run_inference,
            inputs=[prompt_input, system_prompt_box, enable_search, enable_dspy,
                    num_results, search_region, temperature, max_tokens,
                    top_p, top_k, rep_penalty, chatbot],
            outputs=[chatbot, pipeline_output],
        )
        prompt_input.submit(
            run_inference,
            inputs=[prompt_input, system_prompt_box, enable_search, enable_dspy,
                    num_results, search_region, temperature, max_tokens,
                    top_p, top_k, rep_penalty, chatbot],
            outputs=[chatbot, pipeline_output],
        )

        clear_chat_btn.click(lambda: ([], ""), outputs=[chatbot, pipeline_output])
        new_session_btn.click(new_session_action, outputs=[chatbot, pipeline_status])

        # Loglar
        refresh_logs_btn.click(get_logs_display,  [log_type, session_only_cb], [log_display])
        clear_logs_btn.click(clear_logs_action,   [clear_table],               [clear_status])

        # Sistem
        refresh_stats_btn.click(get_stats_display, outputs=[stats_display])

        # İlk yükleme
        demo.load(refresh_models,      outputs=[model_dropdown, load_status])
        demo.load(get_stats_display,   outputs=[stats_display])

    return demo


# ===================== MAIN =====================

if __name__ == "__main__":
    logger.info("OpenVINO LLM Studio başlatılıyor...")
    (ROOT / "logs").mkdir(exist_ok=True)

    demo = build_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        inbrowser=True,
        share=False,
        theme=THEME,
        css=CSS,
    )
