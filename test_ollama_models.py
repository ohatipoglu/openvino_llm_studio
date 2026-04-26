#!/usr/bin/env python
"""
Test script to verify Ollama model listing works correctly.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

print("=" * 60)
print("Testing Ollama Model Listing")
print("=" * 60)

# Test 1: Direct Ollama API call
print("\n[Test 1] Direct Ollama API call...")
try:
    import requests
    r = requests.get('http://localhost:11434/api/tags', timeout=3)
    print(f"  Status: {r.status_code}")
    if r.ok:
        models = r.json().get('models', [])
        print(f"  Models found: {len(models)}")
        for m in models:
            print(f"    - {m['name']} ({m.get('details', {}).get('parameter_size', '?')})")
    else:
        print(f"  Error: {r.text}")
except Exception as e:
    print(f"  ERROR: {e}")

# Test 2: Through Orchestrator
print("\n[Test 2] Through Orchestrator (ollama backend)...")
try:
    from core.orchestrator import Orchestrator
    orch = Orchestrator()
    
    # Set backend to ollama
    orch.set_backend("ollama")
    
    # Get model choices
    choices = orch.get_model_choices()
    print(f"  Models found: {len(choices)}")
    for c in choices:
        print(f"    - {c}")
        
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Through OllamaBackend directly
print("\n[Test 3] Through OllamaBackend directly...")
try:
    from modules.ipex_backend import OllamaBackend
    ollama = OllamaBackend()
    
    print(f"  Ollama available: {ollama.is_available()}")
    print(f"  Backend type: {ollama.detect_backend_type()}")
    
    models = ollama.list_models()
    print(f"  Models found: {len(models)}")
    for m in models:
        print(f"    - {m.model_id} ({m.size_gb:.1f} GB)")
        
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Test Complete")
print("=" * 60)
