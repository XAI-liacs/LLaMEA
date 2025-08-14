#!/usr/bin/env python3
import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

# exp-08-08_033835-LLaMEA-qwen2.5-coder_14b-ELA-basins_scaled_seperable_scaled
EXP_DIR_RE = re.compile(
    r"^exp-(?P<dt>\d{2}-\d{2}_\d{6})-LLaMEA-(?P<llm>.+?)-ELA-(?P<problem>.+)$"
)

def parse_dt_token(token: str) -> Optional[datetime]:
    """Parse '08-08_033835' as MM-DD_HHMMSS; assume this year (UTC), try ±1y for year rollovers."""
    now = datetime.now(timezone.utc)
    for year in (now.year, now.year - 1, now.year + 1):
        try:
            dt = datetime.strptime(f"{year}-{token}", "%Y-%m-%d_%H%M%S")
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None

def fs_times(path: Path) -> Tuple[datetime, datetime]:
    stat = path.stat()
    start = datetime.fromtimestamp(getattr(stat, "st_ctime", stat.st_mtime), tz=timezone.utc)
    end = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
    return start, end

def to_iso(dt: Optional[datetime]) -> Optional[str]:
    return None if dt is None else dt.astimezone(timezone.utc).isoformat()

def safe_json_loads(line: str) -> Optional[dict]:
    try:
        return json.loads(line)
    except Exception:
        return None

def extract_from_jsonl_meta(log_path: Path) -> Tuple[Optional[datetime], Optional[datetime], Dict[str, Any]]:
    """
    Extract meta fields if they ever appear; your sample logs typically don't have timestamps/seed/budget/evaluations.
    We still scan first/last entries for 'seed', 'budget', 'evaluations'.
    """
    meta = {"seed": None, "budget": None, "evaluations": None}
    if not log_path.exists():
        return None, None, meta

    try:
        with log_path.open("r", encoding="utf-8", errors="ignore") as f:
            lines = [ln for ln in f if ln.strip()]
    except Exception:
        lines = []

    first = safe_json_loads(lines[0]) if lines else None
    last = safe_json_loads(lines[-1]) if lines else None

    def maybe_num(d: dict, key: str):
        if d is None or key not in d or meta[key] is not None:
            return
        try:
            meta[key] = int(d[key])
        except Exception:
            try:
                meta[key] = float(d[key])
            except Exception:
                pass

    for d in (first, last):
        for k in ("seed", "budget", "evaluations"):
            maybe_num(d, k)

    # no timestamps in your sample format — return None for caller fallback
    return None, None, meta

def count_jsonl_records(log_path: Path) -> int:
    if not log_path.exists():
        return 0
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        return sum(1 for ln in f if ln.strip())

def parse_fitness(val) -> Optional[float]:
    try:
        f = float(val)
        if f != f:  # NaN
            return None
        return f
    except Exception:
        return None

def best_solution_from_log(log_path: Path) -> Optional[dict]:
    """
    Return the JSON object (solution) with the maximal numeric 'fitness'.
    If none suitable, return the last valid JSON object.
    """
    if not log_path.exists():
        return None
    best = None
    best_fit = None
    last_valid = None
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            obj = safe_json_loads(ln)
            if not isinstance(obj, dict):
                continue
            last_valid = obj
            fit = parse_fitness(obj.get("fitness"))
            if fit is not None and (best_fit is None or fit > best_fit):
                best_fit = fit
                best = obj
    return best or last_valid

def reshape(root: Path, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    # Collect exp dirs
    entries = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        m = EXP_DIR_RE.match(p.name)
        if not m:
            continue
        dt = parse_dt_token(m.group("dt")) or datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)

        entries.append({
            "path": p,
            "dt": dt,
            "llm": m.group("llm"),
            "problem": m.group("problem"),
        })

    # Sort by increasing timestamp
    entries.sort(key=lambda x: x["dt"])

    counters: Dict[tuple, int] = {}
    runs = []
    global_starts, global_ends = [], []

    # Prepare experimentlog.jsonl
    exp_log_path = outdir / "experimentlog.jsonl"
    with exp_log_path.open("w", encoding="utf-8") as exp_log:
        for e in entries:
            key = (e["llm"], e["problem"])
            counters[key] = counters.get(key, 0) + 1
            counter = counters[key]

            run_dir_name = f"run-{e['llm']}-{e['problem']}-{counter}"
            run_dir = outdir / run_dir_name
            run_dir.mkdir(parents=True, exist_ok=True)

            src_log = e["path"] / "log.jsonl"
            dst_log = run_dir / "log.jsonl"

            if src_log.exists():
                shutil.copy2(src_log, dst_log)
            else:
                dst_log.touch()

            # Extract minimal meta; fall back to FS times
            start_ts, end_ts, meta = extract_from_jsonl_meta(src_log)
            if start_ts is None or end_ts is None:
                fs_s, fs_e = (fs_times(src_log) if src_log.exists() else fs_times(e["path"]))
                start_ts = start_ts or fs_s
                end_ts = end_ts or fs_e

            # Derive evaluations & budget if missing
            evaluations = meta.get("evaluations")
            if evaluations is None:
                evaluations = count_jsonl_records(src_log) if src_log.exists() else 0
            budget = meta.get("budget")
            if budget is None:
                budget = evaluations            # Seed increases per (llm, problem) combo, starting at 0
            seed = counter - 1

            # Build per-run dict for progress.json
            problem = e["problem"]
            extra_method = ""
            if problem.endswith("-sharing"):
                problem = problem.replace("-sharing", "")
                extra_method = "-sharing"
            run_entry = {
                "method_name": f"LLaMEA-{e['llm']}{extra_method}",
                "problem_name": e["problem"],
                "seed": seed,
                "budget": budget,
                "evaluations": evaluations,
                "start_time": to_iso(start_ts),
                "end_time": to_iso(end_ts),
                "log_dir": run_dir_name,
            }
            runs.append(run_entry)
            global_starts.append(start_ts)
            global_ends.append(end_ts)

            # Build experimentlog.jsonl entry (structure from your sample)
            best_sol = best_solution_from_log(src_log) if src_log.exists() else None
            exp_record = {
                "method_name": f"LLaMEA-{e['llm']}{extra_method}",
                "problem_name": problem.replace("_scaled", ""),
                "llm_name": e["llm"],
                "method": {
                    "method_name": f"LLaMEA-{e['llm']}{extra_method}",
                    "budget": budget,
                    "kwargs": {},
                },
                "problem": {
                    "name": e["problem"],
                },
                "llm": {
                    "model": e["llm"],
                    "code_pattern": "",
                    "name_pattern": "",
                    "desc_pattern": "",
                    "cs_pattern": "",
                },
                "solution": best_sol or {},
                "log_dir": run_dir_name,
                "seed": seed,
            }
            exp_log.write(json.dumps(exp_record, ensure_ascii=False) + "\n")

    # Write progress.json
    if runs:
        overall_start = min(global_starts)
        overall_end = max(global_ends)
    else:
        now = datetime.now(timezone.utc)
        overall_start = overall_end = now

    progress = {
        "start_time": to_iso(overall_start),
        "end_time": to_iso(overall_end),
        "current": len(runs),
        "total": len(runs),
        "runs": runs,
    }

    with (outdir / "progress.json").open("w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)

    print(f"Done. {len(runs)} runs -> {outdir}")
    print(f"experimentlog.jsonl: {exp_log_path}")

def main():
    parser = argparse.ArgumentParser(description="Reshape LLaMEA experiment logs from multiple folders into a single Experiment folder compatible with BLADE loggers.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Directory containing exp-* folders")
    parser.add_argument("--out", type=Path, default=Path("../../BLADE/results/ELA_experiment"), help="Output directory")
    args = parser.parse_args()
    reshape(args.root.resolve(), args.out.resolve())

if __name__ == "__main__":
    main()
