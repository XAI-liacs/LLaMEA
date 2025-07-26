import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from llamea.solution import Solution


def _random_tournament(solutions):
    scores = {s.id: 0 for s in solutions}
    for i in range(len(solutions)):
        for j in range(i + 1, len(solutions)):
            winner = np.random.choice([solutions[i], solutions[j]])
            scores[winner.id] += 1
    for s in solutions:
        s.set_scores(scores[s.id])
    return solutions


def evaluate_tournament(solutions, scenarios=None, game_dir=None, timeout=600):
    """Evaluate a group of solutions in a Planet Wars tournament.

    This function expects the Planet Wars RTS environment to be available.
    Each solution is saved to a temporary file and then the external
    tournament runner is invoked. The resulting win counts are parsed and
    stored in the solution's fitness attribute. If the environment is not
    available, a random tournament is used as fallback.
    """
    if game_dir is None:
        # Fallback to random tournament if no environment is provided
        return _random_tournament(solutions)

    tmp = Path(tempfile.mkdtemp())
    agent_paths = []
    for sol in solutions:
        agent_file = tmp / f"{sol.name}_{sol.id}.py"
        agent_file.write_text(sol.code)
        agent_paths.append(agent_file)

    cmd = ["./gradlew", "runEvaluation"]
    if scenarios:
        cmd += [f"--args={' '.join(map(str, scenarios))}"]
    try:
        subprocess.run(cmd, cwd=game_dir, check=True, timeout=timeout)
        # Expect results in game_dir/results/sample/league.md
        result_file = Path(game_dir) / "results" / "sample" / "league.md"
        if result_file.exists():
            text = result_file.read_text()
            for sol in solutions:
                m = re.search(rf"{sol.name}.*?(\d+\.\d+)", text)
                if m:
                    sol.set_scores(float(m.group(1)))
    except Exception:
        return _random_tournament(solutions)
    finally:
        for p in agent_paths:
            p.unlink(missing_ok=True)
        tmp.rmdir()
    return solutions
