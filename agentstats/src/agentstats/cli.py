"""Command line interface: ``agentstats report|compare|plan|audit``."""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="agentstats", description="Statistics for agent benchmarks")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("report", help="markdown report for a run table")
    r.add_argument("runs"); r.add_argument("--alpha", type=float, default=0.05)
    r.add_argument("--factors", nargs="*", default=None)

    c = sub.add_parser("compare", help="test agent A vs agent B")
    c.add_argument("runs"); c.add_argument("a"); c.add_argument("b")
    c.add_argument("--level", choices=["task", "run"], default="task")
    c.add_argument("--inference", choices=["benchmark", "population"], default="benchmark")
    c.add_argument("--strata", nargs="*", default=[])

    p = sub.add_parser("plan", help="minimum runs from a pilot run table")
    p.add_argument("pilot"); p.add_argument("--delta", type=float, required=True,
                                            help="smallest gap to detect, fraction (0.02 = 2pp)")
    p.add_argument("--alpha", type=float, default=0.05); p.add_argument("--power", type=float, default=0.8)
    p.add_argument("--method", choices=["task", "run", "max"], default="max")
    p.add_argument("--all-pairs", action="store_true")

    g = sub.add_parser("plan-generic", help="minimum runs without a pilot")
    g.add_argument("--tasks", type=int, required=True); g.add_argument("--delta", type=float, required=True)
    g.add_argument("--flakiness", type=float, default=0.05)
    g.add_argument("--alpha", type=float, default=0.05); g.add_argument("--power", type=float, default=0.8)

    a = sub.add_parser("audit", help="audit a published leaderboard (JSON {agent: score_pp})")
    a.add_argument("scores"); a.add_argument("--sd", type=float, required=True,
                                             help="single-run SD in pp (e.g. from a pilot)")

    w = sub.add_parser("swebench-audit", help="audit the SWE-bench leaderboard from a clone of SWE-bench/experiments")
    w.add_argument("repo"); w.add_argument("--split", default="verified")
    w.add_argument("--top", type=int, default=20); w.add_argument("--pass1", action="store_true")
    w.add_argument("--sd", type=float, default=1.2, help="single-run SD in pp")

    args = ap.parse_args(argv)
    import agentstats as ast

    if args.cmd == "report":
        print(ast.report(ast.load(args.runs), alpha=args.alpha, factors=args.factors))
    elif args.cmd == "compare":
        res = ast.compare(ast.load(args.runs), args.a, args.b, level=args.level,
                          inference=args.inference, strata=args.strata)
        print(json.dumps(res.to_dict(), indent=2))
    elif args.cmd == "plan":
        out = ast.plan(ast.load(args.pilot), args.delta, alpha=args.alpha, power=args.power,
                       method=args.method, adjacent_only=not args.all_pairs)
        print(out.to_markdown(index=False, floatfmt=".4g"))
        print(f"\nRun every agent at least {out.k_min.max():g} times.")
    elif args.cmd == "plan-generic":
        k = ast.k_min_generic(args.tasks, args.delta, args.flakiness, alpha=args.alpha, power=args.power)
        print(f"k_min = {k:g} runs per agent")
    elif args.cmd == "audit":
        with open(args.scores) as f:
            scores = json.load(f)
        print(ast.audit_leaderboard(scores, args.sd).to_markdown(index=False, floatfmt=".3g"))
    elif args.cmd == "swebench-audit":
        runs, meta = ast.load_swebench_submissions(args.repo, args.split, pass_at_1_only=args.pass1)
        top = meta.head(args.top)
        names = list(top.submission)
        df = runs[runs.agent.isin(names)]
        adj = ast.pairwise(df, names, adjacent_only=True, inference="population", n_perm=20000)
        tr = ast.tiers(df, inference="population", n_perm=20000)
        scores = dict(zip(top.submission, 100 * top.n_resolved / runs.task.nunique()))
        au = ast.audit_leaderboard(scores, args.sd)
        st = {m: ast.rerun_stability(scores, args.sd, method=m) for m in ("plug-in", "eb")}
        print(f"# SWE-bench {args.split}: top {len(names)}{' pass@1' if args.pass1 else ''} submissions\n")
        print(adj[["a", "b", "gap", "p_value", "p_holm", "significant"]].to_markdown(index=False, floatfmt=".3g"))
        print(f"\nAdjacent gaps significant (Holm): {int(adj.significant.sum())}/{len(adj)}")
        print(f"Adjacent gaps within single-run noise (SD {args.sd} pp): {int(au.within_noise.sum())}/{len(au)}")
        print(f"Entries tied with #1: {int((tr.tier == 1).sum())}; statistical tiers: {tr.tier.nunique()}")
        for m, v in st.items():
            print(f"Re-run ({m}): P(#1 changes) = {v['p_top1_changes']:.2f}, E[Kendall tau] = {v['kendall_tau']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
