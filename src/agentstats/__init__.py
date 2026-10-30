"""agentstats: run-to-run variance, significance tests and minimum-runs planning
for agent benchmarks (SWE-bench, WebArena, OSWorld, ...).

Quick start::

    import agentstats as ast
    df = ast.load("runs.csv")                     # agent, task, run, success, [factors...]
    ast.agent_scores(df)                          # scores with run-noise CIs
    ast.pairwise(df, adjacent_only=True)          # Holm-corrected paired tests
    ast.tiers(df)                                 # statistically tied groups
    ast.flip_statistics(df)                       # how often single-run leaderboards disagree
    ast.plan(pilot_df, delta=0.02)                # minimum runs per agent for a 2pp gap
"""
from .data import load, task_stats, run_scores, factor_columns, load_swebench_submissions
from .estimate import agent_scores, single_run_sd, flakiness, design_effect
from .compare import compare, pairwise, holm, Comparison
from .power import plan, plan_pair, min_runs_for_exact_run_test, k_min_generic, k_min_from_components, min_detectable_gap, RunPlan
from .rank import rank_distribution, tiers, flip_statistics, single_run_leaderboards, factor_rank_agreement
from .variance import decompose, audit_leaderboard, rerun_stability
from .report import report, check_protocol, PROTOCOL

__version__ = "0.2.0"

__all__ = [
    "load", "task_stats", "run_scores", "factor_columns", "load_swebench_submissions",
    "agent_scores", "single_run_sd", "flakiness", "design_effect",
    "compare", "pairwise", "holm", "Comparison",
    "plan", "plan_pair", "min_runs_for_exact_run_test", "k_min_generic", "k_min_from_components", "min_detectable_gap", "RunPlan",
    "rank_distribution", "tiers", "flip_statistics", "single_run_leaderboards", "factor_rank_agreement",
    "decompose", "audit_leaderboard", "rerun_stability", "report", "check_protocol", "PROTOCOL",
]
