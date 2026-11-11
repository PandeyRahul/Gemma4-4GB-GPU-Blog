# Gemma 4 on a 4GB GPU — code for the blog

| File | What it does |
|---|---|
| `gemma_vram_probe.py` | For Gemma 4 E2B/E4B × context sizes (2K/8K/32K): % of model on GPU (Ollama `/api/ps`), VRAM (`nvidia-smi`), tokens/sec. Appends to `probe_results.json`. |
| `plot_probe.py` | Turns `probe_results.json` into `probe_chart.png`. |
| `vision_test.py` | Ask Gemma 4 to describe an image (e.g. the chart). |
| `Modelfile.gemma4-4gb` | Gemma 4 E2B with a fixed 4K context. |
| `make_images.py` | Rebuilds the concept figures in `img/`. |

## Run

```bash
pip install -r requirements.txt
ollama pull gemma4:e2b
ollama pull gemma4:e4b

# 1. default settings
python gemma_vram_probe.py

# 2. restart Ollama with memory savers, then run again with a label
#    Linux/macOS:  OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve
#    PowerShell:   $env:OLLAMA_FLASH_ATTENTION="1"; $env:OLLAMA_KV_CACHE_TYPE="q8_0"; ollama serve
python gemma_vram_probe.py --label "kv q8_0"

# 3. chart
python plot_probe.py

# vision
python vision_test.py probe_chart.png gemma4:e2b

# custom 4GB model
ollama create gemma4-4gb -f Modelfile.gemma4-4gb
ollama run gemma4-4gb
```

On Windows, quit the Ollama tray app first so your `ollama serve` with the env vars is the one running.
Check exact tags with `ollama list` / https://ollama.com/library/gemma4 and edit `MODELS` in the probe if needed.
