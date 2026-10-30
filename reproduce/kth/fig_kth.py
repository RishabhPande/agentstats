import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
df = pd.read_csv("kth_runs.csv")
common = df.groupby("task").agent.nunique(); common = set(common[common == df.agent.nunique()].index)
runs = df[df.task.isin(common)]
k = runs.groupby("task").run.nunique()
rs = runs.groupby(["agent", "run"]).success.mean().unstack() * 100
rs = rs.loc[rs.mean(axis=1).sort_values().index]
def short(a):
    sc = "nano-agent" if a.startswith("nano") else "R2E-Gym"
    m = "Devstral-2" if "devstral" in a else ("DeepSWE" if "DeepSWE" in a or "deepswe" in a.lower() else "Qwen3-32B")
    return f"{m} · {sc} · " + ("T=0" if "temp0" in a else "default T")
plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                     "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"})
fig, ax = plt.subplots(figsize=(6.2, 3.0))
rng = np.random.default_rng(0)
for y, (a, row) in enumerate(rs.iterrows()):
    col = "#eb6834" if "temp0" in a else "#2a78d6"
    v = row.dropna().to_numpy()
    ax.plot([v.min(), v.max()], [y, y], color=col, lw=1, alpha=0.35)
    ax.scatter(v, y + rng.uniform(-0.12, 0.12, len(v)), s=10, color=col, edgecolor="white", linewidth=0.4, zorder=3)
    ax.plot([v.mean()] * 2, [y - 0.32, y + 0.32], color="#0b0b0b", lw=1.4, zorder=4)
ax.set_yticks(range(len(rs)), [short(a) for a in rs.index])
ax.set_xlabel("% resolved on SWE-bench Verified (one dot per independent run)")
ax.grid(axis="x", color="#e6e5e1", lw=0.5); ax.set_axisbelow(True)
ax.scatter([], [], s=10, color="#2a78d6", label="default temperature"); ax.scatter([], [], s=10, color="#eb6834", label="temperature 0")
ax.plot([], [], color="#0b0b0b", lw=1.4, label="mean of 10 runs"); ax.legend(frameon=False, loc="lower right", fontsize=6.5)
fig.tight_layout(); fig.savefig("fig_kth.pdf")
