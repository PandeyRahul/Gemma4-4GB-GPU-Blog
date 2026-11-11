import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
BG="#fcfcfb"; INK="#0b0b0b"; INK2="#52514e"; MUTED="#a3a29c"; GRID="#e6e5e0"
BLUE="#2a78d6"; ORANGE="#eb6834"; AQUA="#1baf7a"; YEL="#eda100"; RED="#e34948"; VIO="#6250d6"
plt.rcParams.update({"font.family":"DejaVu Sans","text.color":INK,"figure.facecolor":BG,"axes.facecolor":BG})
def box(ax,x,y,w,h,c,r=0.08,ec=None,lw=0):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=0,rounding_size={r}",fc=c,ec=ec or BG,lw=lw or 2))

# 1 HERO
fig,ax=plt.subplots(figsize=(14,7)); ax.axis("off"); ax.set_xlim(0,14); ax.set_ylim(0,7)
ax.text(0.6,6.1,"Gemma 4 on a 4 GB GPU.",fontsize=34,weight="bold")
ax.text(0.6,5.4,"A 5-billion-parameter model that thinks like a 2B one. Does it fit?",fontsize=17,color=INK2)
box(ax,0.6,0.7,12.8,3.9,"#f1f0ec",r=0.25)
ax.text(1.0,4.15,"RTX 3050 · 4 GB VRAM",fontsize=14,color=INK2,weight="bold")
box(ax,1.0,2.4,5.6,1.4,BLUE); ax.text(3.8,3.25,"Gemma 4 E2B",ha="center",fontsize=18,color="white",weight="bold")
ax.text(3.8,2.75,"text · images · audio",ha="center",fontsize=12,color="white")
box(ax,7.0,2.4,5.9,1.4,ORANGE); ax.text(9.95,3.25,"Gemma 4 E4B",ha="center",fontsize=18,color="white",weight="bold")
ax.text(9.95,2.75,"smarter… but bigger",ha="center",fontsize=12,color="white")
ax.text(3.8,1.4,"✓ fits (with a sane context)",ha="center",fontsize=14,color=AQUA,weight="bold")
ax.text(9.95,1.4,"⚠ spills to CPU — how bad?",ha="center",fontsize=14,color=RED,weight="bold")
fig.savefig("img/01_hero.png",dpi=110,bbox_inches="tight",pad_inches=0.3); plt.close()

# 2 PLE: effective vs total params (model card numbers)
fig,ax=plt.subplots(figsize=(13,3.6))
data=[("Gemma 4 E2B",2.3,2.8),("Gemma 4 E4B",4.5,3.5)]
for i,(n,eff,ple) in enumerate(data):
    y=1-i
    ax.barh(y,eff,color=BLUE,height=0.5); ax.barh(y,ple,left=eff,color=MUTED,height=0.5)
    ax.text(eff/2,y,f"{eff}B do the thinking",ha="center",va="center",color="white",fontsize=12,weight="bold")
    ax.text(eff+ple/2,y,f"{ple}B per-layer\nembedding tables",ha="center",va="center",color=INK,fontsize=11)
    ax.text(eff+ple+0.1,y,f"{eff+ple:.1f}B total",va="center",fontsize=12,color=INK2)
ax.set_yticks([1,0],[d[0] for d in data],fontsize=13); ax.set_xticks([]); ax.set_xlim(0,9.4)
for s in ("top","right","bottom","left"): ax.spines[s].set_visible(False)
ax.tick_params(length=0)
ax.set_title("Why it’s called “E2B”: E = effective parameters",loc="left",fontsize=16,weight="bold",pad=12)
fig.text(0.02,-0.04,"Blue = the transformer weights every token goes through. Gray = big lookup tables (one small vector per token per layer): lots of memory, almost no compute.",fontsize=11,color=INK2)
fig.tight_layout(); fig.savefig("img/02_ple.png",dpi=110,bbox_inches="tight",pad_inches=0.3); plt.close()

# 3 VRAM budget concept
fig,ax=plt.subplots(figsize=(14,5)); ax.axis("off"); ax.set_xlim(0,14); ax.set_ylim(0,5)
ax.text(0.3,4.55,"What actually has to fit in 4 GB",fontsize=18,weight="bold")
for row,(title,segs,note) in enumerate([
    ("Short chat  (num_ctx 4096)",[("model weights",5.0,BLUE),("KV cache",1.3,AQUA),("buffers",1.2,YEL)],"fits → 100% GPU, fast"),
    ("Long document  (num_ctx 32768)",[("model weights",5.0,BLUE),("KV cache",3.9,AQUA),("buffers",1.2,YEL)],"doesn’t fit → some layers move to CPU, slow")]):
    y=2.75-row*1.75
    ax.text(0.3,y+0.95,title,fontsize=13,weight="bold",color=INK2)
    x=0.3
    for lab,w,c in segs:
        box(ax,x,y,w,0.7,c,r=0.05); ax.text(x+w/2,y+0.35,lab,ha="center",va="center",color="white" if c!=YEL else INK,fontsize=11,weight="bold"); x+=w
    ax.text(max(x+0.2,8.6),y+0.35,note,va="center",fontsize=12,color=AQUA if row==0 else RED,weight="bold")
ax.plot([8.3,8.3],[0.6,4.1],color=RED,ls="--",lw=2); ax.text(8.35,4.15,"4 GB",color=RED,fontsize=12,weight="bold")
ax.text(0.3,0.2,"Illustration, not to scale: the weights are fixed, but the KV cache grows with the context length you ask for.",fontsize=11,color=INK2)
fig.savefig("img/03_vram_budget.png",dpi=110,bbox_inches="tight",pad_inches=0.3); plt.close()

# 4 CHEAT SHEET
fig,ax=plt.subplots(figsize=(16,6.4)); ax.axis("off"); ax.set_xlim(0,16); ax.set_ylim(0,6.4)
ax.text(0.2,5.95,"Running Gemma 4 on a 4 GB GPU: my checklist",fontsize=24,weight="bold")
rows=[("Setting","Use this","Why"),
      ("Model","gemma4:e2b","E4B usually won’t fit fully on 4 GB"),
      ("Context","num_ctx 4096 (Modelfile)","KV cache grows with context; small = stays on GPU"),
      ("Attention","OLLAMA_FLASH_ATTENTION=1","Less memory for attention, needed for KV quantization"),
      ("KV cache","OLLAMA_KV_CACHE_TYPE=q8_0","Roughly halves KV cache memory vs the default f16"),
      ("Check","ollama ps → 100% GPU","Anything like 60%/40% CPU/GPU means it spilled")]
y=5.15
for i,(a,b,c) in enumerate(rows):
    if i==0:
        for x,t in [(0.2,a),(3.0,b),(8.6,c)]: ax.text(x,y,t,fontsize=12.5,color=INK2,weight="bold")
        ax.plot([0.2,15.8],[y-0.3]*2,color=MUTED,lw=1)
    else:
        ax.text(0.2,y,a,fontsize=12.5,va="center"); ax.text(3.0,y,b,fontsize=12.5,weight="bold",color=BLUE,va="center",family="DejaVu Sans Mono")
        ax.text(8.6,y,c,fontsize=11.5,color=INK2,va="center")
        ax.plot([0.2,15.8],[y-0.42]*2,color=GRID,lw=1)
    y-=0.85 if i else 0.75
fig.savefig("img/04_cheatsheet.png",dpi=110,bbox_inches="tight",pad_inches=0.3); plt.close()
print("ok")
