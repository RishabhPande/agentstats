"""Run tables: the one data format every agentstats function consumes.

A *run table* is a long-format table with one row per (agent, task, run):

    agent   task          run  success  api_date    env_version  latency
    oh-v1   django-11099  0    1        2026-07-01  v1.3         p50
    oh-v1   django-11099  1    0        2026-08-01  v1.3         p95
    ...

Required columns: ``agent``, ``task``, ``run``, ``success``. ``success`` may be
binary (0/1) or a score in [0, 1] (partial credit). Any extra column is
treated as a candidate *nuisance factor* (API snapshot date, tool latency
profile, environment/container version, harness version, hardware, ...).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

REQUIRED = ("agent", "task", "run", "success")


def load(path_or_df, *, agent="agent", task="task", run="run", success="success") -> pd.DataFrame:
    """Load and validate a run table from a CSV/Parquet/JSONL path or a DataFrame."""
    if isinstance(path_or_df, pd.DataFrame):
        df = path_or_df.copy()
    else:
        p = str(path_or_df)
        if p.endswith(".parquet"):
            df = pd.read_parquet(p)
        elif p.endswith((".jsonl", ".ndjson")):
            df = pd.read_json(p, lines=True)
        else:
            df = pd.read_csv(p)
    df = df.rename(columns={agent: "agent", task: "task", run: "run", success: "success"})
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"run table is missing required columns: {missing}")
    df["agent"] = df["agent"].astype(str)
    df["task"] = df["task"].astype(str)
    df["success"] = pd.to_numeric(df["success"], errors="raise").astype(float)
    if ((df["success"] < 0) | (df["success"] > 1)).any():
        raise ValueError("success must lie in [0, 1]")
    dup = df.duplicated(["agent", "task", "run"])
    if dup.any():
        raise ValueError(f"{int(dup.sum())} duplicated (agent, task, run) rows")
    return df.reset_index(drop=True)


def factor_columns(df: pd.DataFrame) -> list[str]:
    """Columns other than the required ones (candidate nuisance factors)."""
    return [c for c in df.columns if c not in REQUIRED]


def is_binary(df: pd.DataFrame) -> bool:
    return bool(np.isin(df["success"].unique(), [0.0, 1.0]).all())


@dataclass
class TaskStats:
    """Per-(agent, task) sufficient statistics, aligned on a common task index."""

    agents: list[str]
    tasks: list[str]
    mean: np.ndarray  # (A, T) mean success per agent-task
    var: np.ndarray   # (A, T) unbiased within-task run variance (nan if k<2)
    k: np.ndarray     # (A, T) number of runs

    def index(self, agent: str) -> int:
        return self.agents.index(agent)


def task_stats(df: pd.DataFrame, agents: Sequence[str] | None = None,
               common_tasks: bool = True) -> TaskStats:
    """Aggregate a run table into per-agent, per-task means / variances / counts.

    With ``common_tasks=True`` (default) only tasks attempted by every selected
    agent are kept, so that all comparisons are paired on the same tasks.
    """
    if agents is not None:
        df = df[df["agent"].isin(agents)]
    g = df.groupby(["agent", "task"])["success"]
    agg = pd.DataFrame({"mean": g.mean(), "var": g.var(ddof=1), "k": g.size()})
    agents_ = list(agents) if agents is not None else sorted(df["agent"].unique())
    if common_tasks:
        per_agent = [set(agg.loc[a].index) for a in agents_]
        tasks = sorted(set.intersection(*per_agent)) if per_agent else []
    else:
        tasks = sorted(df["task"].unique())
    if not tasks:
        raise ValueError("no tasks are shared by all selected agents")
    idx = pd.MultiIndex.from_product([agents_, tasks], names=["agent", "task"])
    agg = agg.reindex(idx)
    shape = (len(agents_), len(tasks))
    return TaskStats(
        agents=agents_,
        tasks=tasks,
        mean=agg["mean"].to_numpy().reshape(shape),
        var=agg["var"].to_numpy().reshape(shape),
        k=agg["k"].fillna(0).to_numpy().reshape(shape).astype(int),
    )


def run_scores(df: pd.DataFrame, by: Iterable[str] = ()) -> pd.DataFrame:
    """Benchmark score of every complete run (mean success over tasks).

    Useful when each ``run`` index is a full pass over the benchmark under one
    condition (seed / API date / env version), as in most leaderboard runs.
    """
    keys = ["agent", "run", *by]
    return df.groupby(keys, as_index=False)["success"].mean().rename(columns={"success": "score"})


def load_swebench_submissions(repo_dir: str, split: str = "verified", pass_at_1_only: bool = False) -> tuple:
    """Load per-instance results of every submission in a local clone of
    https://github.com/SWE-bench/experiments (e.g. ``git clone --depth 1``).

    Returns ``(runs, meta)``: a run table with one run per submission
    (``agent`` = submission folder name) and a metadata DataFrame
    (name, date, n_resolved, attempts, agent scaffold, model). Submissions with
    zero resolved instances are dropped as broken. Requires PyYAML.
    """
    import glob
    import json
    import os
    import yaml

    root = os.path.join(repo_dir, "evaluation", split)
    ids: set = set()
    for f in glob.glob(f"{root}/*/per_instance_details.json"):
        ids |= set(json.load(open(f)))
    rows, meta = [], []
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        res = None
        if os.path.exists(f"{p}/results/results.json"):
            res = set(json.load(open(f"{p}/results/results.json")).get("resolved", []))
        elif os.path.exists(f"{p}/per_instance_details.json"):
            res = {k for k, v in json.load(open(f"{p}/per_instance_details.json")).items() if v.get("resolved")}
        if not res:
            continue
        m = {}
        for mf in ("metadata.yaml", "metadata.yml"):
            if os.path.exists(f"{p}/{mf}"):
                m = yaml.safe_load(open(f"{p}/{mf}")) or {}
        info, tags = m.get("info", {}) or {}, m.get("tags", {}) or {}
        attempts = str((tags.get("system") or {}).get("attempts"))
        if pass_at_1_only and attempts != "1":
            continue
        meta.append(dict(submission=d, date=d[:8], name=info.get("name", d), n_resolved=len(res & ids),
                         attempts=attempts, scaffold=tags.get("agent"),
                         model=",".join(map(str, tags.get("model") or []))))
        rows += [(d, i, 0, float(i in res)) for i in ids]
    runs = pd.DataFrame(rows, columns=["agent", "task", "run", "success"])
    return load(runs), pd.DataFrame(meta).sort_values("n_resolved", ascending=False, ignore_index=True)
