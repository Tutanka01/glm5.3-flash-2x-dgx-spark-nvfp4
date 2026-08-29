#!/usr/bin/env python3
"""Grade an A/B quality artifact from bench-glm53.py (EXL3 vs official API).

The KLD panel in docs/QUALITY.md scores the *weights*; this grader scores the
*end-to-end lane* on identical coding tasks, per the last open promotion item.
Run the comparison with raw content capture, then grade:

    export ZAI_API_KEY=...
    python3 bench-glm53.py \
        --prompts vllm-exl3/tests/ab_quality_prompts.jsonl \
        --runs 1 --thinking off --save-content \
        --compare-base-url 'https://<official API>/v1' \
        --compare-model 'glm-5.3-flash' \
        --output results/ab-quality-<stamp>.json
    python3 vllm-exl3/tests/grade_ab_quality.py \
        --artifact results/ab-quality-<stamp>.json

Graders are deterministic: the two exec-checked tasks run the extracted
function against fixed unit tests in a `python3 -c` subprocess (this executes
model output locally — use --no-exec to skip those), the two JSON tasks are
parsed and compared structurally. A task passes for a target when every
successful run of that target passed (temperature 0 ⇒ runs are repeats).

Exit codes: 0 = graded (see per-task matrix); 2 = artifact unusable (missing
--save-content content, no targets). --self-test validates the graders on
embedded reference solutions.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

FENCE_RE = re.compile(r"```(?:python|py|json)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)

MERGE_TESTS = (
    "assert _norm(merge_intervals([[1,3],[2,6],[8,10],[15,18]])) == [[1,6],[8,10],[15,18]]\n"
    "assert _norm(merge_intervals([[1,4],[4,5]])) == [[1,5]]\n"
    "assert _norm(merge_intervals([[1,4],[0,4]])) == [[0,4]]\n"
    "assert _norm(merge_intervals([[1,4],[2,3]])) == [[1,4]]\n"
    "assert _norm(merge_intervals([])) == []\n"
)
MERGE_DRIVER = (
    "def _norm(seq):\n"
    "    return [list(map(int, pair)) for pair in (seq or [])]\n"
    + MERGE_TESTS
    + "print('AB_QUALITY_PASS')\n"
)

TWOSUM_TESTS = (
    "assert _norm(two_sum_sorted([2,7,11,15], 9)) == [0,1]\n"
    "assert _norm(two_sum_sorted([1,2,3,4,4], 8)) == [3,4]\n"
    "assert _norm(two_sum_sorted([-3,-1,0,2,5], 1)) == [1,3]\n"
    "assert _norm(two_sum_sorted([0,0], 0)) == [0,1]\n"
)
TWOSUM_DRIVER = (
    "def _norm(seq):\n"
    "    return [int(v) for v in seq]\n"
    + TWOSUM_TESTS
    + "print('AB_QUALITY_PASS')\n"
)


def first_python_block(content: str) -> str | None:
    candidates = python_candidates(content)
    return candidates[0] if candidates else None


def python_candidates(content: str) -> list[str]:
    """Fenced block first; otherwise the whole message, then from `def` down."""
    match = FENCE_RE.search(content)
    if match:
        code = match.group(1).strip()
        return [code] if "def " in code else []
    text = content.strip()
    if "def " not in text:
        return []
    candidates = [text]
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("def "):
            from_def = "\n".join(lines[index:]).strip()
            if from_def != candidates[0]:
                candidates.append(from_def)
            break
    return candidates


def parse_json_value(content: str) -> Any | None:
    text = content.strip()
    match = FENCE_RE.search(text)
    if match:
        text = match.group(1).strip()
    for candidate in (text, _balanced_slice(text)):
        if candidate is None:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _balanced_slice(text: str) -> str | None:
    start = min(
        (index for index in (text.find("{"), text.find("[")) if index != -1),
        default=-1,
    )
    end = max(text.rfind("}"), text.rfind("]"))
    if start == -1 or end <= start:
        return None
    return text[start : end + 1]


def run_exec_task(code: str, driver: str, timeout: float) -> tuple[bool, str]:
    program = f"{code}\n\n{driver}"
    try:
        completed = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout:.0f}s"
    if completed.returncode == 0 and "AB_QUALITY_PASS" in completed.stdout:
        return True, "tests passed"
    detail = (completed.stderr or completed.stdout).strip().splitlines()
    return False, detail[-1] if detail else f"exit {completed.returncode}"


def norm_int(value: Any) -> Any:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return value


# ------------------------------- graders -----------------------------------
# Each grader: (content) -> (passed, detail). Graders must be deterministic.


def grade_merge_intervals(content: str, exec_enabled: bool) -> tuple[bool, str]:
    return grade_exec_candidates(content, exec_enabled, MERGE_DRIVER)


def grade_two_sum_sorted(content: str, exec_enabled: bool) -> tuple[bool, str]:
    return grade_exec_candidates(content, exec_enabled, TWOSUM_DRIVER)


def grade_exec_candidates(
    content: str, exec_enabled: bool, driver: str
) -> tuple[bool, str]:
    candidates = python_candidates(content)
    if not candidates:
        return False, "no Python function found"
    if not exec_enabled:
        return True, "skipped (--no-exec): function extracted"
    detail = "no runnable candidate"
    for candidate in candidates:
        passed, detail = run_exec_task(candidate, driver, timeout=20.0)
        if passed:
            return True, detail
    return False, detail


def grade_csv_to_json(content: str, _exec_enabled: bool) -> tuple[bool, str]:
    value = parse_json_value(content)
    if not isinstance(value, list) or len(value) != 2:
        return False, "expected a JSON array of 2 objects"
    expected = [
        {"name": "ada", "role": "infra", "commits": 12},
        {"name": "linus", "role": "kernel", "commits": 7},
    ]
    for got, want in zip(value, expected):
        if not isinstance(got, dict):
            return False, "row is not an object"
        if sorted(got) != sorted(want):
            return False, f"keys {sorted(got)} != {sorted(want)}"
        for key, want_value in want.items():
            got_value = norm_int(got[key]) if key == "commits" else got[key]
            if got_value != want_value:
                return False, f"{key}: {got[key]!r} != {want_value!r}"
    return True, "structure and values match"


def grade_release_config(content: str, _exec_enabled: bool) -> tuple[bool, str]:
    value = parse_json_value(content)
    if not isinstance(value, dict):
        return False, "expected a JSON object"
    if sorted(value) != ["name", "port", "tags"]:
        return False, f"keys {sorted(value)} != ['name', 'port', 'tags']"
    if value["name"] != "glm53":
        return False, f"name: {value['name']!r} != 'glm53'"
    if norm_int(value["port"]) != 8080:
        return False, f"port: {value['port']!r} != 8080"
    if value["tags"] != ["stable", "quantized"]:
        return False, f"tags: {value['tags']!r}"
    return True, "schema and values match"


GRADERS: dict[str, Callable[[str, bool], tuple[bool, str]]] = {
    "merge_intervals": grade_merge_intervals,
    "two_sum_sorted": grade_two_sum_sorted,
    "csv_to_json": grade_csv_to_json,
    "release_config": grade_release_config,
}

REFERENCE_SOLUTIONS = {
    "merge_intervals": "```python\ndef merge_intervals(intervals):\n"
    "    intervals = sorted(intervals)\n"
    "    merged = []\n"
    "    for start, end in intervals:\n"
    "        if merged and start <= merged[-1][1]:\n"
    "            merged[-1][1] = max(merged[-1][1], end)\n"
    "        else:\n"
    "            merged.append([start, end])\n"
    "    return merged\n```",
    "two_sum_sorted": "```python\ndef two_sum_sorted(numbers, target):\n"
    "    left, right = 0, len(numbers) - 1\n"
    "    while left < right:\n"
    "        total = numbers[left] + numbers[right]\n"
    "        if total == target:\n"
    "            return [left, right]\n"
    "        if total < target:\n"
    "            left += 1\n"
    "        else:\n"
    "            right -= 1\n"
    "    return []\n```",
    "csv_to_json": '[{"name": "ada", "role": "infra", "commits": 12}, '
    '{"name": "linus", "role": "kernel", "commits": 7}]',
    "release_config": '{"name": "glm53", "port": 8080, "tags": ["stable", "quantized"]}',
}


def self_test(exec_enabled: bool) -> int:
    for name, grader in GRADERS.items():
        passed, detail = grader(REFERENCE_SOLUTIONS[name], exec_enabled)
        status = "OK " if passed else "FAIL"
        print(f"[grade-ab-quality] self-test {status} {name}: {detail}")
        if not passed:
            return 1
    broken = grade_merge_intervals("def merge_intervals(a):\n    return a", exec_enabled)
    if broken[0]:
        print("[grade-ab-quality] self-test FAIL: bad merge solution passed")
        return 1
    print("[grade-ab-quality] self-test: all graders behave")
    return 0


def grade_artifact(
    artifact: dict[str, Any],
    exec_enabled: bool,
) -> tuple[int, str, dict[str, Any] | None]:
    results = artifact.get("results")
    if not isinstance(results, list) or not results:
        return 2, "artifact has no results[]", None
    if all(not r.get("content") for r in results):
        return 2, "artifact has no raw content — re-run bench-glm53.py with --save-content", None

    targets = sorted({r.get("target", "?") for r in results})
    tasks = sorted({r.get("prompt", "?") for r in results if r.get("prompt") in GRADERS})
    if not tasks:
        return (
            2,
            "no gradable tasks in artifact (expected: " + ", ".join(GRADERS) + ")",
            None,
        )

    lines: list[str] = []
    score: dict[str, int] = {target: 0 for target in targets}
    detail_rows: dict[str, dict[str, Any]] = {}
    for task in tasks:
        grader = GRADERS[task]
        cells: list[str] = []
        detail_rows[task] = {}
        for target in targets:
            runs = [r for r in results if r.get("target") == target and r.get("prompt") == task]
            outcomes: list[dict[str, Any]] = []
            all_passed = bool(runs)
            for run in runs:
                if not run.get("ok"):
                    outcomes.append({"ok": False, "detail": run.get("error") or "request failed"})
                    all_passed = False
                    continue
                passed, detail = grader(run["content"], exec_enabled)
                outcomes.append({"ok": passed, "detail": detail})
                if not passed:
                    all_passed = False
            detail_rows[task][target] = outcomes
            if all_passed:
                score[target] += 1
            marks = "/".join("P" if o["ok"] else "F" for o in outcomes) or "-"
            cells.append(f"{target}={marks}")
        lines.append(f"  {task:<16} " + "  ".join(cells))

    header = f"  {'task':<16} " + "  ".join(f"{t:<{len(t) + 2}}" for t in targets)
    score_line = f"  {'PASS RATE':<16} " + "  ".join(
        f"{score[t]}/{len(tasks)}" for t in targets
    )
    body = "\n".join([header, *lines, "  " + "-" * 40, score_line])
    for target in targets:
        ok_runs = sum(1 for r in results if r.get("target") == target and r.get("ok"))
        body += f"\n  {target}: {ok_runs} ok requests graded"

    graded = {
        "tasks": {task: detail_rows[task] for task in tasks},
        "score": {target: f"{score[target]}/{len(tasks)}" for target in targets},
    }
    return 0, body, graded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--no-exec", action="store_true",
                        help="skip the two exec-checked tasks (JSON tasks only)")
    parser.add_argument("--out", type=Path,
                        help="write the graded detail JSON here")
    parser.add_argument("--self-test", action="store_true")
    options = parser.parse_args()
    if options.self_test:
        return self_test(exec_enabled=not options.no_exec)
    if not options.artifact:
        parser.error("--artifact is required (or use --self-test)")
    try:
        artifact = json.loads(options.artifact.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"grade_ab_quality error: {exc}", file=sys.stderr)
        return 2
    code, report, graded = grade_artifact(artifact, exec_enabled=not options.no_exec)
    print(report)
    if code == 0 and graded is not None:
        if options.out:
            options.out.parent.mkdir(parents=True, exist_ok=True)
            options.out.write_text(
                json.dumps(graded, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            print(f"Wrote {options.out}")
        else:
            print(json.dumps(graded, ensure_ascii=False))
    elif code:
        print(f"grade_ab_quality error: {report}", file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
