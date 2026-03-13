"""
ipex_worker.py  —  KALDIRILDI / DEPRECATED

Bu dosya artık kullanılmamaktadır.

NEDEN KALDIRILDI:
  ipex-llm projesi Mart 2026'da End-of-Life (EOL) ilan edildi.
  Tüm XPU/SYCL optimizasyonları doğrudan PyTorch 2.4+ ana dalına
  aktarıldı (upstream). ipex-llm ve bu worker mimarisi artık
  desteklenmiyor ve bakımı yapılmıyor.

  Kaynak:
    https://github.com/intel-analytics/ipex-llm
    (arşivlenmiş repo — yeni commit yok)

YENİ MİMARİ:
  IPEX TCP socket mimarisinin yerini LlamaCppBackend aldı.
  Artık:
    • Ayrı bir conda environment (ipex_studio) gerekmez.
    • TCP socket sunucusu gerekmez.
    • llama-cpp-python openvino_studio ortamında doğrudan çalışır.

  modules/ipex_backend.py    → LlamaCppBackend (GGUF / SYCL)
  modules/ipex_worker_client.py → IPEXWorkerClient (LlamaCppBackend sarmalayıcı)

INTEL ARC iGPU HIZLANDIRMA:
  İki seçenek:

  1) Ollama + IPEX-LLM Ollama Fork (ÖNERİLEN — kolay kurulum)
     • https://github.com/ipex-llm/ollama/releases
     • İndir: ollama-ipex-llm-2.2.0-win.zip
     • C:\\ipex-ollama\\ altına çıkart
     • start_ollama.bat:
         set OLLAMA_NUM_GPU=999
         set ZES_ENABLE_SYSMAN=1
         set SYCL_CACHE_PERSISTENT=1
         set no_proxy=localhost,127.0.0.1
         ollama.exe serve
     • Beklenen: "Found 1 SYCL devices: Intel Arc Graphics"
     • Sonra: Studio'da Ollama backend seç → iGPU otomatik devreye girer

  2) llama-cpp-python SYCL (ileri düzey — derleme gerektirir)
     • Intel oneAPI Base Toolkit kurulu olmalı
     • Windows:
         set CMAKE_ARGS=-DGGML_SYCL=ON
         pip install llama-cpp-python --no-binary llama-cpp-python
     • Linux:
         CMAKE_ARGS="-DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx" \\
           pip install llama-cpp-python --no-binary llama-cpp-python
     • Doküman: https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md

BU DOSYA HAKKINDA:
  Geriye dönük referans için saklanmıştır.
  Çalıştırmayın — hiçbir işlem yapmaz.
"""

import sys
import logging

logger = logging.getLogger(__name__)


def main():
    logger.error(
        "ipex_worker.py kullanımdan kaldırıldı (ipex-llm EOL, Mart 2026).\n"
        "Yeni mimari için modules/ipex_backend.py ve modules/ipex_worker_client.py "
        "dosyalarına bakın.\n"
        "Intel Arc iGPU hızlandırma için IPEX-LLM Ollama fork veya "
        "llama-cpp-python SYCL kullanın."
    )
    print("\n" + "=" * 70)
    print("  ipex_worker.py KULLANIM DIŞI (ipex-llm EOL, Mart 2026)")
    print("=" * 70)
    print("\n  Yeni backend: modules/ipex_backend.py → LlamaCppBackend")
    print("  Intel Arc iGPU için: Ollama + IPEX-LLM Ollama fork")
    print("    https://github.com/ipex-llm/ollama/releases\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
