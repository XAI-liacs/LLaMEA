
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Union
from llamea import Solution
from pathlib import Path
import subprocess, json, os

PLANETWARS_DIR = "/home/neocortex/repos/planet-wars-rts-python"

GAME_DIR = Path(PLANETWARS_DIR) / "app" / "src" / "main" / "python"
VENV_PY = Path(PLANETWARS_DIR) / ".venv" / "bin" / "python"


def evaluate_tournament(solutions, logger=None, timeout=7200):
    """Evaluate a group of solutions in a Planet Wars tournament.

    This function expects the Planet Wars RTS environment to be available.
    Each solution is saved to a temporary file and then the external
    tournament runner is invoked. The resulting win counts are parsed and
    stored in the solution's fitness attribute. If the environment is not
    available, a random tournament is used as fallback.
    """
    tmp = GAME_DIR / "llamea"
    tmp.mkdir(exist_ok=True)
    agent_paths = []
    agent_files = []
    for sol in solutions:
        agent_file = tmp / f"{sol.id}.py"
        agent_file.write_text(sol.code)
        agent_files.append(agent_file)
        
        # First we check that each agent is fast enough to run in the tournament.
        cmd = [str(VENV_PY), "-m", "runner_utils.ultrafast_agent_eval", "--agent", f"llamea.{sol.id}.{sol.name}"]
        try:
            env = dict(os.environ, PYTHONPATH=str(GAME_DIR))
            subprocess.run(cmd, cwd=GAME_DIR, env=env, check=True, timeout=15)
            result_file = GAME_DIR / "ultrafast_agent_eval_results.json"
            if not result_file.exists():
                # If the result file does not exist, the agent is too slow.
                sol.set_scores(0, feedback=f"Agent {sol.name} is too slow to run in the tournament. Please speed it up.")
                continue
            # (optional) verify the result actually mentions this agent
            data = json.loads(result_file.read_text())
            key = f"llamea.{sol.id}.{sol.name}"
            if key not in data:
                sol.set_scores(0, feedback=f"Agent {sol.name} failed the speed check.")
                continue
            else:
                # check if the agent has a 0 score, which means there was an error loading it.
                if data[key][0] == -1:
                    sol.set_scores(0, feedback=data[key][1])
                    continue
            agent_paths.append(f"llamea.{sol.id}.{sol.name}")
        except subprocess.TimeoutExpired:
            sol.set_scores(0, feedback=f"Agent {sol.name} exceeded the 15s pre-check timeout. Please speed it up.")
            continue
        except Exception as e:
            sol.set_scores(0, feedback=f"Agent {sol.name} failed pre-check: {e}")
            continue
        finally:
            # do not let this file pollute the next iteration
            if result_file.exists():
                result_file.unlink()

    # Now run the tournament with the fast agents.
    cmd = [str(VENV_PY), "-m", "runner_utils.fast_agent_eval_2", "--agent"]
    for a in agent_paths:
        cmd += [a]

    try:
        # ensure modules in app/src/main/python are importable
        env = dict(os.environ, PYTHONPATH=str(GAME_DIR))

        subprocess.run(cmd, cwd=GAME_DIR, env=env, check=True, timeout=timeout)
        # Expect results in GAME_DIR/results/sample/league.md
        result_file = GAME_DIR / "fast_agent_eval_results.json"
        if result_file.exists():
            json_data = json.loads(result_file.read_text())
            for sol in solutions:
                score = json_data.get(f"llamea.{sol.id}.{sol.name}", [-1, f"No data for llamea.{sol.id}.{sol.name}"])
                if score[0] >= 0:
                    sol.set_scores(score[0], feedback=score[1])
                #otherwise the agent was too slow.

    except Exception as e:
        for sol in solutions:
            if sol.feedback == "":
                sol.set_scores(
                    0, feedback="Tournament failed or timed out with exception: " + str(e)
                )
    finally:
        for p in agent_files:
            p.unlink(missing_ok=True)

    return solutions


def _iter_jsonl(path: Union[str, "Path"]) -> Iterable[Dict[str, Any]]:
    with open(path, "r") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj
            except Exception:
                # skip bad lines
                continue

def best_solutions_from_log(
    log_path: Union[str, "Path"],
    *,
    id_key: str = "id",
    name_key: str = "name",
    code_key: str = "code",
    fitness_key: str = "fitness",
    generation_key: str = "generation",
) -> List[Any]:
    """Parse a JSONL log and return a list with the best solution per generation.

    - Chooses the entry with the highest `fitness` for each `generation`.
    - Skips rows missing code/generation/fitness, or with non-numeric fitness.
    - Tie-breaker: keeps the first seen entry with the max fitness.
    """
    best_by_gen: Dict[int, Dict[str, Any]] = {}
    for row in _iter_jsonl(log_path):
        if generation_key not in row or code_key not in row or fitness_key not in row:
            continue
        try:
            gen = int(row[generation_key])
            fit = float(row[fitness_key])
        except Exception:
            continue
        # If this generation is new or this row has better fitness, store it
        cur = best_by_gen.get(gen)
        if cur is None or fit > float(cur.get(fitness_key, float("-inf"))):
            best_by_gen[gen] = row

    results: List[Any] = []
    for gen in sorted(best_by_gen.keys()):
        r = best_by_gen[gen]
        ctor_kwargs = {
            "id": str(r.get(id_key, "")),
            "name": str(r.get(name_key, "")),
            "code": r.get(code_key, ""),
            "fitness": float(r.get(fitness_key)),
            "generation": int(r.get(generation_key)),
            "description": r.get("description"),
            "parent_ids": r.get("parent_ids"),
            "operator": r.get("operator"),
            "metadata": r.get("metadata"),
            "feedback": r.get("feedback"),
            "error": r.get("error"),
            "configspace": r.get("configspace"),
        }
        results.append(Solution.from_dict(ctor_kwargs))
    return results

best = best_solutions_from_log("/home/neocortex/repos/LLaMEA/exp-08-22_072838-LLaMEA-multi-llm-planetwars-sharing/log.jsonl")#"/home/neocortex/repos/LLaMEA/exp-08-08_185234-LLaMEA-gpt-5-2025-08-07-planetwars/log2.jsonl")
print(best)
print("Evaluating large tournament...")
solution_scores = evaluate_tournament(best)
for sol in solution_scores:
    print(f"Solution {sol.id} ({sol.name}): {sol.fitness} - {sol.feedback}")
    # write to csv file
    with open("solution_scores.csv", "a") as f:
        f.write(f"{sol.id},{sol.name},{sol.fitness},{sol.feedback}\n") 
    