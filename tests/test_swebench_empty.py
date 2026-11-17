"""Local SWE-bench fixtures, including filters that select no submissions."""
import json
import os
import subprocess
import sys

import pytest

from agentstats.data import load_swebench_submissions

META_COLUMNS = ["submission", "date", "name", "n_resolved", "attempts", "scaffold", "model"]


def fixture(root, name, resolved=("t1",), attempts="1", metadata=True):
    directory = root / name
    directory.mkdir()
    (directory / "per_instance_details.json").write_text(json.dumps({
        t: {"resolved": t in resolved} for t in ("t1", "t2")
    }))
    if metadata:
        (directory / "metadata.yaml").write_text(
            f"info:\n  name: {name}\ntags:\n  system:\n    attempts: '{attempts}'\n"
        )


@pytest.mark.parametrize("scenario", ["empty", "zero_resolved", "filtered", "no_metadata"])
def test_no_eligible_submissions_returns_empty_tables_with_columns(tmp_path, scenario):
    root = tmp_path / "evaluation" / "verified"
    root.mkdir(parents=True)
    if scenario == "zero_resolved":
        fixture(root, "20260101_zero", resolved=())
    elif scenario == "filtered":
        fixture(root, "20260101_multi", attempts="5")
    elif scenario == "no_metadata":
        fixture(root, "20260101_unknown", metadata=False)
    runs, meta = load_swebench_submissions(str(tmp_path), pass_at_1_only=True)
    assert runs.empty and meta.empty
    assert list(runs.columns) == ["agent", "task", "run", "success"]
    assert list(meta.columns) == META_COLUMNS


@pytest.mark.parametrize("scenario", ["empty", "zero_resolved", "filtered"])
def test_empty_audit_reports_no_eligible_submissions_without_traceback(tmp_path, scenario):
    root = tmp_path / "evaluation" / "verified"
    root.mkdir(parents=True)
    if scenario == "zero_resolved":
        fixture(root, "20260101_zero", resolved=())
    elif scenario == "filtered":
        fixture(root, "20260101_multi", attempts="5")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath("src") + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run([sys.executable, "-m", "agentstats.cli", "swebench-audit",
                             str(tmp_path), "--pass1"], capture_output=True, text=True, env=env)
    assert result.returncode == 2
    assert "no eligible SWE-bench submissions" in result.stderr
    assert "Traceback" not in result.stderr
    assert result.stdout == ""


def test_nonempty_filter_preserves_rows_counts_and_sorting(tmp_path):
    root = tmp_path / "evaluation" / "verified"
    root.mkdir(parents=True)
    fixture(root, "20260101_one", resolved=("t1",))
    fixture(root, "20260102_two", resolved=("t1", "t2"))
    fixture(root, "20260103_filtered", attempts="5")
    fixture(root, "20260104_zero", resolved=())
    runs, meta = load_swebench_submissions(str(tmp_path), pass_at_1_only=True)
    assert meta.submission.tolist() == ["20260102_two", "20260101_one"]
    assert meta.n_resolved.tolist() == [2, 1]
    assert len(runs) == 4 and runs.success.sum() == 3
    assert set(runs.agent) == set(meta.submission)
    assert set(runs.task) == {"t1", "t2"}
