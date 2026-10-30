"""Sensitivity of the SWE-bench audit to the assumed single-run SD, plug-in vs empirical-Bayes."""
import json, numpy as np, pandas as pd
from scipy.stats import kendalltau, norm
m = pd.read_csv("swe_meta.csv"); m = m[m.n_resolved > 0]
boards = {"top20_pass1": m[m.attempts.astype(str) == "1"].nlargest(20, "n_resolved"),
          "mini_v2": m[m.submission.str.startswith("20260217_mini-v2.0.0")].sort_values("n_resolved", ascending=False)}
rng = np.random.default_rng(1)
out = []
for lab, b in boards.items():
    s = b.n_resolved.to_numpy() / 500
    gaps = -np.diff(s)
    for sd_pp in (0.8, 1.2, 1.8):
        sd = sd_pp / 100
        within = int(np.sum(gaps < norm.ppf(.975) * np.sqrt(2) * sd))
        for method in ("plug-in", "emp. Bayes"):
            N = 4000
            if method == "plug-in":
                theta = np.tile(s, (N, 1))
            else:
                mu, v_obs = s.mean(), s.var(ddof=1)
                B = max(0.0, (v_obs - sd ** 2) / v_obs)
                theta = mu + B * (s - mu) + rng.normal(0, np.sqrt(B) * sd, (N, len(s)))
            A = theta + rng.normal(0, sd, theta.shape); C = theta + rng.normal(0, sd, theta.shape)
            out.append(dict(board=lab, sd_pp=sd_pp, method=method, within_noise=within, n_gaps=len(gaps),
                            p_top1_changes=float(np.mean(A.argmax(1) != C.argmax(1))),
                            p_true_best_not_first=float(np.mean(A.argmax(1) != theta.argmax(1))),
                            kendall_tau=float(np.mean([kendalltau(x, y).statistic for x, y in zip(A[:1500], C[:1500])]))))
df = pd.DataFrame(out); print(df.to_string())
df.to_csv("sensitivity.csv", index=False)
d = json.load(open("swe_audit.json")); d["sensitivity"] = out; json.dump(d, open("swe_audit.json", "w"), indent=1, default=float)
