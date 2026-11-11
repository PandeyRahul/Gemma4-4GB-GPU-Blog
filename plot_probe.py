"""Turn probe_results.json into probe_chart.png (speed + GPU share per context size)."""
import json
import matplotlib.pyplot as plt

rows = json.load(open("probe_results.json"))
series = sorted({(r["model"], r["label"]) for r in rows})
colors = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for c, (model, label) in zip(colors, series):
    rs = sorted([r for r in rows if r["model"] == model and r["label"] == label], key=lambda r: r["num_ctx"])
    x = [str(r["num_ctx"]) for r in rs]
    name = model if label == "default" else f"{model} ({label})"
    axes[0].plot(x, [r["tok_per_s"] for r in rs], marker="o", lw=2, color=c, label=name)
    axes[1].plot(x, [r["gpu_share"] * 100 for r in rs], marker="o", lw=2, color=c, label=name)
axes[0].set_title("Generation speed (tokens/sec, higher is better)", loc="left", weight="bold")
axes[1].set_title("Share of the model on the GPU (%)", loc="left", weight="bold")
axes[1].set_ylim(0, 105)
for ax in axes:
    ax.set_xlabel("context length (num_ctx)")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e6e5e0")
axes[0].legend(frameon=False)
fig.tight_layout(); fig.savefig("probe_chart.png", dpi=130)
print("saved probe_chart.png")
