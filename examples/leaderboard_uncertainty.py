"""How much does a single-run leaderboard tell you? Real data: 12 agent configurations x 10 runs
x 500 SWE-bench Verified tasks (reproduce/kth/kth_runs.csv, ASSERT-KTH, CC-BY-4.0).

    python examples/leaderboard_uncertainty.py [out.png]
"""
import sys, warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import agentstats as ast

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
out = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "docs" / "leaderboard_uncertainty.png")

df = ast.load(str(ROOT / "reproduce" / "kth" / "kth_runs.csv"))
sc = ast.agent_scores(df).sort_values("score", ascending=False).reset_index(drop=True)
tiers = ast.tiers(df, n_perm=2000).set_index("agent")["tier"]
runs = df.groupby(["agent", "run"]).success.mean().unstack() * 100
leaders = runs.idxmax(axis=0)                       # who is #1 in each single run
share = leaders.value_counts().reindex(sc.agent, fill_value=0) / runs.shape[1] * 100
def short(a):
    t0 = a.endswith(("__temp0", "-temp0"))
    a = a.replace("__temp0", "").replace("-temp0", "")
    scaffold, model = ("nano-agent", a[11:]) if a.startswith("nano-agent") else ("r2e-gym", a[8:])
    model = model.replace("agentica-org_", "").replace("agentica-org__", "").replace("Qwen_", "").replace("mistral_", "")
    model = model.lstrip("_-")
    return f"{scaffold} | {model.lower()}" + (" | temp 0" if t0 else "")
palette = ["#2e86ab", "#e4572e", "#59a14f", "#b07aa1", "#edc948", "#76b7b2"]

fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 5), dpi=140, gridspec_kw={"width_ratios": [1.6, 1]}, sharey=True)
y = np.arange(len(sc))[::-1]
for yi, r in zip(y, sc.itertuples()):
    a.scatter(runs.loc[r.agent].values, [yi] * runs.shape[1], s=14, color="#aaa", zorder=1)
    a.errorbar(r.score * 100, yi, xerr=[[(r.score - r.lo) * 100], [(r.hi - r.score) * 100]], fmt="o",
               color=palette[(int(tiers[r.agent]) - 1) % len(palette)], capsize=3, zorder=2)
a.set_yticks(y); a.set_yticklabels([short(s) for s in sc.agent], fontsize=8)
a.set_xlabel("SWE-bench Verified pass rate (%)")
a.set_title("Mean, 95% CI, each of 10 runs (grey); colour = statistical tier", fontsize=8, loc="left")
b.barh(y, share.values, color="#e4572e")
b.set_xlabel("% of single runs where it ranks #1")
b.set_title("If you had published one run", fontsize=8, loc="left")
for ax in (a, b):
    ax.grid(axis="x", alpha=.25)
fig.suptitle(f"{runs.shape[1]} identical runs of {len(sc)} configurations: {leaders.nunique()} different agents took first place",
             fontsize=11, x=.01, ha="left")
fig.text(.01, .005, "Real data: ASSERT-KTH/agentic-evals-artifacts (CC-BY-4.0). Tiers from agentstats.tiers (Holm-corrected).",
         fontsize=7, color="#666")
fig.tight_layout(rect=(0, .03, 1, .95))
Path(out).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out)
print(f"{leaders.nunique()} distinct #1 over {runs.shape[1]} runs; tiers={tiers.nunique()}; wrote {out}")
