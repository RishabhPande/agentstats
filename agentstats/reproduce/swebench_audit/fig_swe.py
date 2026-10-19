import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = json.load(open("swe_audit.json")); m = pd.read_csv("swe_meta.csv")
sd = d["rerun_sim_from_published"]["top20_pass1"]["sd_pp"]
plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                     "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"})
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4), gridspec_kw={"width_ratios": [1, 1], "wspace": 0.75})
for ax, lb, title in [(axes[0], d["leaderboards"][0], "Top-20 single-attempt submissions"),
                      (axes[1], d["leaderboards"][2], "mini-SWE-agent v2.0.0, 11 models")]:
    names = sorted(set(lb["top_tier"]) | set(m.submission), key=lambda s: 0)  # placeholder
    sub = m.set_index("submission")
    if lb is d["leaderboards"][0]:
        rows = m[m.attempts.astype(str) == "1"].nlargest(20, "n_resolved")
    else:
        rows = m[m.submission.str.startswith("20260217_mini-v2.0.0") & (m.n_resolved > 0)].sort_values("n_resolved", ascending=False)
    y = np.arange(len(rows))[::-1]
    s = rows.n_resolved.to_numpy() / 5
    top = rows.submission.isin(lb["top_tier"]).to_numpy()
    for yi, si, t in zip(y, s, top):
        ax.plot([si - 1.96 * sd, si + 1.96 * sd], [yi, yi], color="#2a78d6" if t else "#b4b3ad", lw=1.6, solid_capstyle="round")
        ax.plot(si, yi, "o", ms=4, color="#2a78d6" if t else "#8a8984", mec="white", mew=0.6)
    lab = [(str(n)[:25] + "…") if len(str(n)) > 26 else str(n) for n in rows.name]
    ax.set_xlim(s.min() - 4, s.max() + 4)
    ax.set_yticks(y, lab); ax.set_xlabel("% resolved (SWE-bench Verified)")
    ax.set_title(title, fontsize=7.5, loc="left")
    ax.grid(axis="x", color="#e6e5e1", lw=0.5); ax.set_axisbelow(True)
fig.subplots_adjust(left=0.235, right=0.99, bottom=0.12, top=0.93)
fig.savefig("fig_swe_audit.pdf")
