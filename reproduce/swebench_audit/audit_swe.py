import json
import numpy as np, pandas as pd
import agentstats as ast

m = pd.read_csv("swe_meta.csv")
runs = pd.read_csv("swe_runs.csv")
m = m[m.n_resolved > 0]
out = {}

# ---------- near-replicates: same pinned model, same scaffold family, pass@1
pairs = [("20250807_mini-v1.7.0_gpt-5-mini", "20260217_mini-v2.0.0_gpt-5-mini"),
         ("20250929_mini-v1.13.3_sonnet-4-5-20250929", "20260217_mini-v2.0.0_claude-4-5-sonnet-high")]
W = runs.pivot(index="task", columns="agent", values="success")
rep = []
for a, b in pairs:
    x, y = W[a], W[b]
    dis = float((x != y).mean())
    phi = dis / 2  # E[1{y1!=y2}] = 2 p(1-p) for iid runs
    rep.append(dict(a=a, b=b, score_a=100 * x.mean(), score_b=100 * y.mean(), gap_pp=100 * (x.mean() - y.mean()),
                    tasks_disagree=int((x != y).sum()), flakiness_upper=phi,
                    implied_single_run_sd_pp=100 * np.sqrt(phi / 500)))
out["replicates"] = rep
print(pd.DataFrame(rep).to_string())

def leaderboard(sub, label, sds=(1.0, 1.5, 2.0)):
    sub = sub.sort_values("n_resolved", ascending=False)
    names = list(sub.submission)
    df = runs[runs.agent.isin(names)]
    adj = ast.pairwise(df, names, adjacent_only=True, inference="population", n_perm=20000)
    res = dict(label=label, n=len(names),
               adjacent_gaps_pp=[round(100 * g, 2) for g in adj.gap],
               median_gap_pp=float(100 * np.median(adj.gap)),
               exact_ties=int((adj.gap.abs() < 1e-9).sum()),
               pop_sig_unadj=int((adj.p_value < .05).sum()), pop_sig_holm=int(adj.significant.sum()))
    scores = dict(zip(sub.submission, 100 * sub.n_resolved / 500))
    for sd in sds:
        au = ast.audit_leaderboard(scores, sd)
        res[f"within_noise_sd{sd}"] = int(au.within_noise.sum())
        res[f"mean_pflip_sd{sd}"] = float(au.p_flip_single_rerun.mean())
    tr = ast.tiers(df, inference="population", n_perm=20000)
    res["tiers_population"] = int(tr.tier.nunique())
    res["top_tier_size"] = int((tr.tier == 1).sum())
    res["top_tier"] = list(tr[tr.tier == 1].agent)
    print(json.dumps({k: v for k, v in res.items() if k not in ("adjacent_gaps_pp", "top_tier")}, indent=1))
    return res, adj

p1 = m[m.attempts.astype(str) == "1"]
r1, adj1 = leaderboard(p1.nlargest(20, "n_resolved"), "Top-20 pass@1 submissions")
r2, adj2 = leaderboard(m.nlargest(20, "n_resolved"), "Top-20 all submissions")
mini = m[m.submission.str.startswith("20260217_mini-v2.0.0")]
r3, adj3 = leaderboard(mini, "mini-SWE-agent v2.0.0 (same scaffold, same date)")
out["leaderboards"] = [r1, r2, r3]
adj1.to_csv("adj_top20_pass1.csv", index=False); adj3.to_csv("adj_mini.csv", index=False)
json.dump(out, open("swe_audit.json", "w"), indent=1, default=float)
