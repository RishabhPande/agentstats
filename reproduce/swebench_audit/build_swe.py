"""Build a per-instance table of all SWE-bench Verified submissions (public, SWE-bench/experiments)."""
import json, glob, os, sys, yaml
import pandas as pd
ROOT = os.path.join(sys.argv[1] if len(sys.argv) > 1 else "experiments", "evaluation", "verified")
ids = set()
for f in glob.glob(f"{ROOT}/*/per_instance_details.json"):
    ids |= set(json.load(open(f)))
rows, meta = [], []
for d in sorted(os.listdir(ROOT)):
    p = os.path.join(ROOT, d)
    res = None
    if os.path.exists(f"{p}/results/results.json"):
        res = set(json.load(open(f"{p}/results/results.json"))["resolved"])
    elif os.path.exists(f"{p}/per_instance_details.json"):
        x = json.load(open(f"{p}/per_instance_details.json"))
        res = {k for k, v in x.items() if v.get("resolved")}
    if res is None:
        continue
    m = {}
    for mf in ("metadata.yaml", "metadata.yml"):
        if os.path.exists(f"{p}/{mf}"):
            m = yaml.safe_load(open(f"{p}/{mf}")) or {}
    info, tags = m.get("info", {}) or {}, m.get("tags", {}) or {}
    system = tags.get("system", {}) or {}
    meta.append(dict(submission=d, date=d[:8], name=info.get("name", d), resolved_reported=info.get("resolved"),
                     n_resolved=len(res & ids) if ids else len(res), attempts=system.get("attempts"),
                     checked=tags.get("checked"), agent=tags.get("agent"), model=",".join(map(str, tags.get("model") or [])),
                     os_system=tags.get("os_system"), extra_ids=len(res - ids)))
    for i in ids:
        rows.append((d, i, int(i in res)))
print(len(ids), "instances;", len(meta), "submissions")
pd.DataFrame(meta).to_csv("swe_meta.csv", index=False)
pd.DataFrame(rows, columns=["agent", "task", "success"]).assign(run=0).to_csv("swe_runs.csv", index=False)
