"""Gemma 4 E2B/E4B can see images. Quick test: describe a picture.

    python vision_test.py path/to/image.png [model]
"""
import sys, ollama

img = sys.argv[1]
model = sys.argv[2] if len(sys.argv) > 2 else "gemma4:e2b"
r = ollama.chat(model=model, options={"num_ctx": 4096},
                messages=[{"role": "user",
                           "content": "Describe this image in 3 bullet points. If it is a chart, read the numbers.",
                           "images": [img]}])
print(r["message"]["content"])
