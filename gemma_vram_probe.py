"""How far can Gemma 4 E2B / E4B go on a small GPU?

For every model x context size it measures:
  - how much of the model Ollama put on the GPU (size_vram / size from /api/ps)
  - real VRAM used (nvidia-smi)
  - generation speed (tokens/sec)
Results are appended to probe_results.json; run plot_probe.py afterwards.

Usage:
    ollama pull gemma4:e2b
    ollama pull gemma4:e4b
    python gemma_vram_probe.py
    # try again with a quantized KV cache (restart `ollama serve` with these set):
    #   OLLAMA_FLASH_ATTENTION=1  OLLAMA_KV_CACHE_TYPE=q8_0
    python gemma_vram_probe.py --label "kv q8_0"
"""
import json, os, subprocess, sys, time
import requests

OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODELS = ["gemma4:e2b", "gemma4:e4b"]          # check `ollama list` for your exact tags
CONTEXTS = [2048, 8192, 32768]
PROMPT = "Explain, in about 150 words, why small language models are useful on laptops."


def vram_mb():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
                                       "--format=csv,noheader,nounits"], text=True)
        return int(out.splitlines()[0])
    except Exception:
        return None


def unload_all():
    for m in requests.get(f"{OLLAMA}/api/ps").json().get("models", []):
        requests.post(f"{OLLAMA}/api/generate", json={"model": m["name"], "keep_alive": 0})
    time.sleep(2)


def gpu_share(model):
    for m in requests.get(f"{OLLAMA}/api/ps").json().get("models", []):
        if m["name"] == model or m["model"] == model:
            return m["size_vram"] / m["size"], m["size"] / 1e9
    return None, None


def run(model, ctx):
    opts = {"num_ctx": ctx, "temperature": 0, "num_predict": 200, "seed": 1}
    requests.post(f"{OLLAMA}/api/generate",               # warm-up / load
                  json={"model": model, "prompt": "hi", "stream": False,
                        "options": {**opts, "num_predict": 1}}).raise_for_status()
    share, total_gb = gpu_share(model)
    used = vram_mb()
    speeds = []
    for _ in range(3):
        r = requests.post(f"{OLLAMA}/api/generate",
                          json={"model": model, "prompt": PROMPT, "stream": False,
                                "options": opts}).json()
        speeds.append(r["eval_count"] / (r["eval_duration"] / 1e9))
    return {"model": model, "num_ctx": ctx, "loaded_gb": round(total_gb, 2),
            "gpu_share": round(share, 3), "vram_mb": used,
            "tok_per_s": round(sum(speeds) / len(speeds), 1)}


if __name__ == "__main__":
    label = sys.argv[sys.argv.index("--label") + 1] if "--label" in sys.argv else "default"
    rows = []
    for model in MODELS:
        for ctx in CONTEXTS:
            unload_all()
            row = run(model, ctx) | {"label": label}
            rows.append(row)
            print(f"{model:12s} ctx={ctx:>6} | loaded {row['loaded_gb']:.2f} GB | "
                  f"{row['gpu_share']*100:5.1f}% on GPU | VRAM {row['vram_mb']} MB | "
                  f"{row['tok_per_s']:6.1f} tok/s")
    path = "probe_results.json"
    data = json.load(open(path)) if os.path.exists(path) else []
    json.dump(data + rows, open(path, "w"), indent=2)
    print(f"saved -> {path}")
