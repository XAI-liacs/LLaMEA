import os
import json
from llamea import Gemini_LLM, OpenAI_LLM, LLaMEA, Multi_LLM
from pathlib import Path
import subprocess, json, os

PLANETWARS_DIR = "/home/neocortex/repos/planet-wars-rts-python"

GAME_DIR = Path(PLANETWARS_DIR) / "app" / "src" / "main" / "python"
VENV_PY = Path(PLANETWARS_DIR) / ".venv" / "bin" / "python"


def evaluate_tournament(solutions, parents, logger=None, timeout=3600):
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
    # include parents in the tournament

    if parents:
        for parent in parents:
            if parent.fitness > 0.0:
                agent_file = tmp / f"{parent.id}.py"
                agent_file.write_text(parent.code)
                agent_files.append(agent_file)
                agent_paths.append(f"llamea.{parent.id}.{parent.name}")
    for sol in solutions:
        agent_file = tmp / f"{sol.id}.py"
        agent_file.write_text(sol.code)
        agent_files.append(agent_file)
        
        # First we check that each agent is fast enough to run in the tournament.
        cmd = [str(VENV_PY), "-m", "runner_utils.ultrafast_agent_eval", "--agent", f"llamea.{sol.id}.{sol.name}"]
        try:
            env = dict(os.environ, PYTHONPATH=str(GAME_DIR))
            subprocess.run(cmd, cwd=GAME_DIR, env=env, check=True, timeout=30)
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
            if parents:
                for parent in parents:
                    score = json_data.get(f"llamea.{parent.id}.{parent.name}", [-1, f"No data for llamea.{parent.id}.{parent.name}"])
                    if score[0] >= 0:
                        parent.set_scores(score[0], feedback=score[1])

    except Exception as e:
        for sol in solutions:
            if sol.feedback == "":
                sol.set_scores(
                    0, feedback="Tournament failed or timed out with exception: " + str(e)
                )
    finally:
        for p in agent_files:
            p.unlink(missing_ok=True)

    return solutions, parents


def evaluate_individual(sol, logger=None, timeout=3600):
    tmp = GAME_DIR / "llamea"
    tmp.mkdir(exist_ok=True)
    agent_paths = []
    agent_files = []
    # do not include parents in the tournament

    solutions = [sol]
    for sol in solutions:
        agent_file = tmp / f"{sol.id}.py"
        agent_file.write_text(sol.code)
        agent_files.append(agent_file)
        
        # First we check that each agent is fast enough to run in the tournament.
        cmd = [str(VENV_PY), "-m", "runner_utils.ultrafast_agent_eval", "--agent", f"llamea.{sol.id}.{sol.name}"]
        try:
            env = dict(os.environ, PYTHONPATH=str(GAME_DIR))
            subprocess.run(cmd, cwd=GAME_DIR, env=env, check=True, timeout=30)
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
            sol.set_scores(0, feedback=f"Agent {sol.name} exceeded the 30s pre-check timeout. Please speed it up.")
            continue
        except Exception as e:
            sol.set_scores(0, feedback=f"Agent {sol.name} failed pre-check: {e}")
            continue
        finally:
            # do not let this file pollute the next iteration
            if result_file.exists():
                result_file.unlink()

    # Now run the tournament with the fast agents per agent

    # first set a default score:
    for sol in solutions:
        if sol.feedback == "":
            sol.set_scores(
                0, feedback="Tournament failed or timed out."
            )
    for a in agent_paths:
        cmd = [str(VENV_PY), "-m", "runner_utils.fast_agent_eval_2", "--agent"]
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
            pass

    #cleanup
    for p in agent_files:
        p.unlink(missing_ok=True)

    return sol

def evaluate_tournament_individually(solutions, parents, logger=None, timeout=3600):
    """Evaluate a group of solutions in a Planet Wars tournament, but in isolation against Galactic armada.

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
    # do not include parents in the tournament


    for sol in solutions:
        agent_file = tmp / f"{sol.id}.py"
        agent_file.write_text(sol.code)
        agent_files.append(agent_file)
        
        # First we check that each agent is fast enough to run in the tournament.
        cmd = [str(VENV_PY), "-m", "runner_utils.ultrafast_agent_eval", "--agent", f"llamea.{sol.id}.{sol.name}"]
        try:
            env = dict(os.environ, PYTHONPATH=str(GAME_DIR))
            subprocess.run(cmd, cwd=GAME_DIR, env=env, check=True, timeout=60)
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

    # Now run the tournament with the fast agents per agent

    # first set a default score:
    for sol in solutions:
        if sol.feedback == "":
            sol.set_scores(
                0, feedback="Tournament failed or timed out."
            )
    for a in agent_paths:
        cmd = [str(VENV_PY), "-m", "runner_utils.fast_agent_eval_2", "--agent"]
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
            pass

    #cleanup
    for p in agent_files:
        p.unlink(missing_ok=True)

    return solutions, parents

role_prompt = (
    "You are an expert game AI developer specialised in real-time strategy games."
)

# Detailed game information provided to the LLM so it can reason about the
# Planet Wars environment.  This describes the data structures that an agent
# receives and how an action should be returned.
game_info = """
Planet Wars is written in Python.  Each turn your agent is given a
`GameState` instance from `core.game_state`.  Important classes are:

- `PlanetWarsPlayer`: base class for agents.  `prepare_to_play_as` sets
  `self.player` (of enum type `Player`) and `self.params` (a `GameParams`
  object with constants such as `transporter_speed` and `max_ticks`).
- `GameState` contains a list of `Planet` objects in `game_state.planets` and
  the current tick number in `game_state.game_tick`.
- `Planet` has attributes `id`, `owner`, `n_ships`, `position` (a `Vec2d` with
  `x` and `y` fields), `growth_rate`, `radius` and optionally `transporter` if a
  fleet is travelling from that planet.
- `Action` represents an order and is created with `Action(player_id,
  source_planet_id, destination_planet_id, num_ships)`.  You may also return
  `Action.do_nothing()` or the constant `Action.DO_NOTHING` to skip a turn.

Example to access planets owned by the player and choose a target:
```python
my_planets = [p for p in game_state.planets
              if p.owner == self.player and p.transporter is None]
enemy_planets = [p for p in game_state.planets
                 if p.owner == self.player.opponent()]
```
Distances can be computed with `p1.position.distance(p2.position)`.

Be aware that Action (and some other classes) are pydantic models, so pass everything by keyword and not positional arguments.
In addition, the agent should react fast, so avoid too many complex computations.
The following Python libraries are available in the environment:
- numpy
- scipy

"""

# Task prompt that combines the detailed game description with the actual
# request for an agent implementation.
task_prompt = (
    game_info
    + "\nDesign a novel Python agent for the Planet Wars RTS game. The agent should"
    + " inherit from `agents.planet_wars_agent.PlanetWarsPlayer` and implement"
    + " the `get_action` and `get_agent_type` methods. The agent should perform robustly"
    + " across different game setups."
)

feedback_prompts = [
    "Refine and simplify the selected agent in order to improve its strategy (and give it a distinct one-line description and distinct name).",
    "Improve the agent by refining its logic. Give it a distinct one-line description and distinct name.",
    "Design a completely new agent that works differently from the ones you have tried before. Give it a distinct one-line description and distinct name.",
    "Improve the planning component (forward simulation) of the agent. Give it a distinct one-line description and distinct name.",
]

example_prompt = """"The initial selected agent for the Planet Wars RTS game, is as follows:

```python
import math
from typing import Dict, List, Optional, Tuple

from agents.planet_wars_agent import PlanetWarsPlayer
from core.game_state import GameState, Action, Player, Planet


class FrontierSentryHyperGuardPlus(PlanetWarsPlayer):
    \"\"\"
    FrontierSentryHyperGuardPlus:
    - Adaptive safety reserves (reduce when holding many planets) to avoid over-defensiveness in dense maps.
    - Opportunistic snipes: capture weak enemy planets (or ones recently launched from) from nearby donors.
    - Slightly more aggressive expansion bias and larger source pools when many planets exist.
    - Keeps original defense-first design with lightweight forward projection.
    \"\"\"

    def get_agent_type(self) -> str:
        return "FrontierSentryHyperGuardPlus"

    # -------------------------- small utilities --------------------------

    def _speed(self) -> float:
        s = float(getattr(self.params, "transporter_speed", 1.0) or 1.0)
        return max(1e-6, s)

    def _dist(self, a: Planet, b: Planet) -> float:
        return float(a.position.distance(b.position))

    def _eta(self, a: Planet, b: Planet) -> float:
        return self._dist(a, b) / self._speed()

    def _clamp_int(self, x: float, lo: int, hi: int) -> int:
        return int(max(lo, min(hi, int(x))))

    def _phase(self, game_state: GameState) -> float:
        mt = float(getattr(self.params, "max_ticks", 1.0) or 1.0)
        return min(1.0, float(game_state.game_tick) / max(1.0, mt))

    def _safety_reserve(self, p: Planet, phase: float) -> int:
        \"\"\"
        Adaptive safety reserve: base reserve increases slightly with phase and growth,
        but is reduced if we own many planets to avoid being overly passive.
        \"\"\"
        base = 5 + int(7 * phase)
        # scale down base if many owned planets — encourages expansion when holding many bases
        my_count = int(getattr(self, "_my_count", 1) or 1)
        # factor shrinks from 1.0 toward 0.5 as my_count grows (~8+)
        factor = max(0.5, 1.0 - 0.06 * max(0, my_count - 1))
        base = max(2, int(base * factor))
        # keep a bit proportional to growth but also scale down if growth is large (to allow aggressive capture)
        extra = int(1.8 * p.growth_rate)
        return base + extra

    def _min_dist_to_set(self, p: Planet, targets: List[Planet]) -> float:
        if not targets:
            return 1e9
        return min(self._dist(p, t) for t in targets)

    # -------------------------- projection model --------------------------

    def _incoming_to_planet(self, game_state: GameState, target_id: int) -> List[Tuple[float, Player, int]]:
        \"\"\"
        Collect incoming fleets to target_id as (eta_ticks, owner, ships) by scanning all planets'
        transporter fields (at most 1 per planet in this environment).
        \"\"\"
        inc: List[Tuple[float, Player, int]] = []
        spd = self._speed()
        # Some environments store planets indexed by id in a list; be defensive when resolving ids.
        id_map = {p.id: p for p in game_state.planets}
        for src in game_state.planets:
            tr = getattr(src, "transporter", None)
            if tr is None:
                continue
            dest_id = getattr(tr, "destination_planet_id", None)
            if dest_id is None:
                dest = getattr(tr, "destination", None)
                dest_id = getattr(dest, "id", None) if dest is not None else None
            if dest_id != target_id:
                continue

            ships = int(getattr(tr, "num_ships", getattr(tr, "n_ships", 0)) or 0)
            owner = getattr(tr, "owner", None)
            if owner is None:
                owner = src.owner

            t_rem = getattr(tr, "ticks_remaining", None)
            if t_rem is None:
                d = float(getattr(tr, "distance", None) or (self._dist(src, id_map[target_id]) if target_id in id_map else 0.0))
                t_rem = d / spd if spd > 0 else 0.0
            inc.append((float(t_rem), owner, ships))
        inc.sort(key=lambda x: x[0])
        return inc

    def _incoming_to_planet_for_index(self, game_state: GameState, pid: int) -> List[Tuple[float, Player, int]]:
        return self._incoming_to_planet(game_state, pid)

    def _project_planet_at_eta(self, game_state: GameState, planet: Planet, eta: float) -> Tuple[Player, float]:
        \"\"\"
        Lightweight projection of a planet at time 'eta' using growth and at most one known incoming transporter.
        Returns (owner_at_eta, ships_at_eta).
        \"\"\"
        owner = planet.owner
        ships = float(planet.n_ships)

        if owner != Player.Neutral:
            ships += float(planet.growth_rate) * float(eta)

        # For simplicity, ignore complex multi-incoming overlaps here beyond growth + known in-flight.
        return owner, ships

    def _simulate_contested_owner(self, game_state: GameState, planet: Planet, horizon: float) -> Tuple[Player, float, float]:
        \"\"\"
        Lightweight event simulation with growth + incoming fleets (if any) up to horizon.
        Returns (owner_end, ships_end, min_margin_owned).
        \"\"\"
        owner = planet.owner
        ships = float(planet.n_ships)
        t = 0.0
        min_margin_owned = float("inf")

        incoming = self._incoming_to_planet(game_state, planet.id)
        incoming = [(ti, ow, sh) for (ti, ow, sh) in incoming if ti <= horizon + 1e-9]

        def grow(dt: float) -> None:
            nonlocal ships
            if owner != Player.Neutral:
                ships += float(planet.growth_rate) * dt

        for (ti, ow, sh) in incoming:
            dt = max(0.0, ti - t)
            grow(dt)
            t = ti

            if ow == owner:
                ships += float(sh)
            else:
                ships -= float(sh)
                if ships < 0:
                    ships = -ships
                    owner = ow

            if owner == self.player:
                min_margin_owned = min(min_margin_owned, ships)

        grow(max(0.0, horizon - t))
        if owner == self.player:
            min_margin_owned = min(min_margin_owned, ships)

        if min_margin_owned == float("inf"):
            min_margin_owned = -1e9
        return owner, ships, min_margin_owned

    # -------------------------- action selection --------------------------

    def get_action(self, game_state: GameState) -> Action:
        me = self.player
        opp = self.player.opponent()

        my_planets: List[Planet] = [
            p for p in game_state.planets
            if p.owner == me and getattr(p, "transporter", None) is None and p.n_ships > 0
        ]
        enemy_planets: List[Planet] = [p for p in game_state.planets if p.owner == opp]
        neutral_planets: List[Planet] = [p for p in game_state.planets if p.owner == Player.Neutral]

        # track my count for adaptive reserves
        self._my_count = len(my_planets)

        if not my_planets:
            return Action.DO_NOTHING

        phase = self._phase(game_state)
        speed = self._speed()

        reference_targets = enemy_planets if enemy_planets else neutral_planets
        frontline: Optional[Planet] = None
        if reference_targets:
            frontline = min(my_planets, key=lambda p: self._min_dist_to_set(p, reference_targets))

        possible_targets = enemy_planets + neutral_planets
        d_scale = 1.0
        if possible_targets:
            sample_src = my_planets[0]
            d_scale = max(1.0, sum(self._dist(sample_src, t) for t in possible_targets) / len(possible_targets))

        # 1) urgent defense (unchanged but uses adaptive reserves)
        defense_horizon = 0.60 * (d_scale / speed) + 6.0
        threatened: List[Tuple[float, Planet, float]] = []  # (urgency, planet, deficit)
        for p in my_planets:
            owner_end, ships_end, min_margin = self._simulate_contested_owner(game_state, p, defense_horizon)
            if owner_end != me or min_margin < 2.0:
                deficit = max(0.0, 6.0 - max(min_margin, 0.0))
                dist_to_enemy = self._min_dist_to_set(p, enemy_planets) if enemy_planets else 1.0
                urgency = deficit + 3.0 * (1.0 - min(1.0, dist_to_enemy / max(1.0, d_scale)))
                threatened.append((urgency, p, deficit))

        if threatened:
            threatened.sort(key=lambda x: x[0], reverse=True)
            _, target_def, deficit = threatened[0]

            donors = []
            for d in my_planets:
                if d.id == target_def.id:
                    continue
                surplus = int(d.n_ships) - self._safety_reserve(d, phase)
                if surplus <= 0:
                    continue
                eta = self._eta(d, target_def)
                donors.append((eta, -surplus, d, surplus))
            if donors:
                donors.sort(key=lambda x: (x[0], x[1]))
                eta, _, donor, surplus = donors[0]
                send = min(surplus, int(math.ceil(deficit + 2.0)))
                if send <= 0:
                    send = min(surplus, max(8, int(0.25 * donor.n_ships)))
                if send > 0:
                    return Action(
                        player_id=me,
                        source_planet_id=donor.id,
                        destination_planet_id=target_def.id,
                        num_ships=int(send),
                    )

        # 1.5) Opportunistic snipes: if enemy planet is weak or recently launched from, try quick nearby capture.
        # This helps in dense maps (more_planets) and against mid-game skirmishes.
        if possible_targets:
            # Build a quick avail map to check donors
            quick_avail: Dict[int, int] = {}
            for s in my_planets:
                a = int(s.n_ships) - self._safety_reserve(s, phase)
                if a > 0:
                    quick_avail[s.id] = a
            if quick_avail:
                # examine enemy planets first for snipes
                snipe_candidates = enemy_planets + [p for p in neutral_planets if p.growth_rate >= 2]
                for ep in sorted(snipe_candidates, key=lambda p: (p.n_ships, p.growth_rate)):
                    # if ep has very low ships or may have recently launched (transporter exists on its planets)
                    low_thresh = max(6, int(1.2 * ep.growth_rate) + 2)
                    nearby_my = sorted(my_planets, key=lambda s: self._eta(s, ep))
                    if not nearby_my:
                        continue
                    nearest = nearby_my[0]
                    eta = self._eta(nearest, ep)
                    # require reasonably close
                    if eta > max(2.5, 0.9 * (d_scale / max(1.0, speed))):
                        continue
                    # Prefer sniping low-defended enemy planets (including ones they launched from)
                    incoming_enemy = self._incoming_to_planet(game_state, ep.id)
                    opponent_launched = any(tr_owner == opp for (_, tr_owner, _) in incoming_enemy)
                    # if planet weak or opponent just launched (source weakened)
                    if ep.n_ships <= low_thresh or opponent_launched:
                        avail = quick_avail.get(nearest.id, 0)
                        if avail <= 0:
                            continue
                        # requirement to capture: current ships + potential growth until arrival
                        extra_growth = ep.growth_rate * eta if ep.owner != Player.Neutral else 0.0
                        req = int(math.ceil(ep.n_ships + extra_growth + 1.5))
                        send = min(avail, req)
                        # don't send almost all of donor's ships: ensure donor keeps some margin
                        donor_reserve = self._safety_reserve(nearest, phase)
                        if int(nearest.n_ships) - send < donor_reserve:
                            send = max(0, int(nearest.n_ships) - donor_reserve)
                        if send > 0:
                            return Action(
                                player_id=me,
                                source_planet_id=nearest.id,
                                destination_planet_id=ep.id,
                                num_ships=int(send),
                            )

        # 2) plan attacks (single or 2-source)
        avail: Dict[int, int] = {}
        for s in my_planets:
            a = int(s.n_ships) - self._safety_reserve(s, phase)
            if a > 0:
                avail[s.id] = a

        if not avail or not possible_targets:
            return self._cohesion_shuffle(my_planets, reference_targets, frontline, d_scale, phase) or Action.DO_NOTHING

        best: Optional[Tuple[float, Tuple[Planet, int], Optional[Tuple[Planet, int]], Planet]] = None

        # if many planets, be slightly more expansionist
        expand_bias = (1.08 - 0.60 * phase) if self._my_count > 6 else (1.05 - 0.65 * phase)

        # use larger source pool when we own many planets
        max_sources = 6 if self._my_count > 6 else 4

        for tgt in possible_targets:
            K = max_sources if len(my_planets) > max_sources else len(my_planets)
            srcs_sorted = sorted(my_planets, key=lambda s: self._eta(s, tgt))[:K]

            min_eta = self._eta(srcs_sorted[0], tgt) if srcs_sorted else 0.0
            owner_at_min, ships_at_min, _ = self._simulate_contested_owner(game_state, tgt, min_eta)
            base_def = ships_at_min
            if owner_at_min != tgt.owner:
                base_def += 0.75 * tgt.growth_rate

            front_bonus = 0.0
            if frontline is not None and possible_targets:
                front_bonus = 0.15 * (1.0 - min(1.0, self._dist(tgt, frontline) / max(1.0, d_scale)))

            if tgt.owner == Player.Neutral:
                value = expand_bias * (1.35 * tgt.growth_rate + 0.25) + front_bonus
            else:
                value = (1.95 * tgt.growth_rate + 0.65) + 0.28 * (tgt.n_ships < 15) + front_bonus

            distance_pen = 0.5 * (min_eta / max(1.0, d_scale / speed))

            margin = 1.06 if tgt.owner == Player.Neutral else 1.16
            req_min = int(math.ceil(max(0.0, base_def) * margin)) + 1

            best_single: Optional[Tuple[float, Planet, int]] = None
            for s in srcs_sorted:
                a = avail.get(s.id, 0)
                if a <= 0:
                    continue
                eta = self._eta(s, tgt)
                extra = 0.0
                if tgt.owner != Player.Neutral:
                    extra = max(0.0, eta - min_eta) * float(tgt.growth_rate)
                req = int(math.ceil(req_min + extra))
                if req <= 0 or req > a:
                    continue
                # send a robust but not full chunk
                send = min(a, max(1, int(0.72 * s.n_ships)))
                # small penalty for using large sending; encourage efficient hits
                cost_pen = 0.011 * send
                eta_pen = 0.22 * (eta / max(1.0, d_scale / speed))
                score = value - distance_pen - eta_pen - cost_pen
                if best_single is None or score > best_single[0]:
                    best_single = (score, s, int(send))

            best_pair: Optional[Tuple[float, Planet, int, Planet, int]] = None
            if len(srcs_sorted) >= 2:
                for i in range(min(5, len(srcs_sorted))):
                    for j in range(i + 1, min(6, len(srcs_sorted))):
                        s1, s2 = srcs_sorted[i], srcs_sorted[j]
                        a1, a2 = avail.get(s1.id, 0), avail.get(s2.id, 0)
                        if a1 <= 0 or a2 <= 0:
                            continue

                        eta1, eta2 = self._eta(s1, tgt), self._eta(s2, tgt)
                        eta_max = max(eta1, eta2)
                        owner_p, ships_p, _ = self._simulate_contested_owner(game_state, tgt, eta_max)
                        def_p = ships_p
                        margin2 = 1.06 if tgt.owner == Player.Neutral else 1.15
                        req_total = int(math.ceil(max(0.0, def_p) * margin2)) + 1

                        if a1 + a2 < req_total:
                            continue

                        cap1 = max(1, int(0.75 * s1.n_ships))
                        cap2 = max(1, int(0.75 * s2.n_ships))
                        # allocate slightly more to faster arrival
                        portion = 0.55 if eta1 <= eta2 else 0.45
                        send1 = min(a1, cap1, int(math.ceil(req_total * portion)))
                        send2 = min(a2, cap2, req_total - send1)
                        if send2 <= 0:
                            continue
                        if send1 + send2 < req_total:
                            need = req_total - (send1 + send2)
                            add1 = min(a1 - send1, cap1 - send1, need)
                            send1 += max(0, add1)
                            need = req_total - (send1 + send2)
                            add2 = min(a2 - send2, cap2 - send2, need)
                            send2 += max(0, add2)
                        if send1 + send2 < req_total:
                            continue

                        cost_pen = 0.0075 * (send1 + send2)
                        eta_pen = 0.20 * (eta_max / max(1.0, d_scale / speed))
                        coord_bonus = 0.045
                        score = value - distance_pen - eta_pen - cost_pen + coord_bonus

                        if best_pair is None or score > best_pair[0]:
                            best_pair = (score, s1, int(send1), s2, int(send2))

            pick_score = None
            pick = None
            if best_single is not None:
                pick_score = best_single[0]
                pick = ((best_single[1], best_single[2]), None)
            if best_pair is not None and (pick_score is None or best_pair[0] > pick_score):
                pick_score = best_pair[0]
                pick = ((best_pair[1], best_pair[2]), (best_pair[3], best_pair[4]))

            if pick_score is None or pick is None:
                continue

            if best is None or pick_score > best[0]:
                best = (pick_score, pick[0], pick[1], tgt)

        if best is not None:
            score, (s1, n1), pair2, tgt = best
            # be slightly more willing to attack when holding many planets
            threshold = (0.12 - 0.04 * phase) - (0.02 if self._my_count > 6 else 0.0)
            if score > threshold and n1 > 0:
                if pair2 is not None:
                    s2, n2 = pair2
                    if self._eta(s2, tgt) < self._eta(s1, tgt):
                        s1, n1 = s2, n2
                return Action(
                    player_id=me,
                    source_planet_id=s1.id,
                    destination_planet_id=tgt.id,
                    num_ships=int(n1),
                )

        # 3) cohesion shuffle
        shuffle = self._cohesion_shuffle(my_planets, reference_targets, frontline, d_scale, phase)
        if shuffle is not None:
            return shuffle

        return Action.DO_NOTHING

    def _cohesion_shuffle(
        self,
        my_planets: List[Planet],
        reference_targets: List[Planet],
        frontline: Optional[Planet],
        d_scale: float,
        phase: float,
    ) -> Optional[Action]:
        \"\"\"Move surplus from deep backline to frontline to keep pressure.\"\"\"
        if frontline is None or len(my_planets) < 2 or not reference_targets:
            return None

        front_d = self._min_dist_to_set(frontline, reference_targets)
        backline: List[Planet] = []
        for p in my_planets:
            if p.id == frontline.id:
                continue
            pd = self._min_dist_to_set(p, reference_targets)
            if pd > front_d + 0.30 * d_scale:
                backline.append(p)
        if not backline:
            return None

        def donor_key(p: Planet) -> Tuple[int, float]:
            surplus = int(p.n_ships) - self._safety_reserve(p, self._phase_placeholder())
            return (surplus, self._min_dist_to_set(p, reference_targets))

        # choose best donor that has surplus and is deep
        donor = max(backline, key=donor_key)
        surplus = int(donor.n_ships) - self._safety_reserve(donor, self._phase_placeholder())
        if surplus <= 8:
            return None

        send = int(min(surplus, max(10, int(0.28 * donor.n_ships))))
        return Action(
            player_id=self.player,
            source_planet_id=donor.id,
            destination_planet_id=frontline.id,
            num_ships=int(send),
        )

    def _phase_placeholder(self) -> float:
        \"\"\"
        Helper used in cohesion shuffle where get_action previously passed phase.
        If _my_count is set, we attempt to approximate phase from params; otherwise default 0.0.
        \"\"\"
        # In practice this function should be fast and conservative.
        try:
            mt = float(getattr(self.params, "max_ticks", 1.0) or 1.0)
            # we don't have direct game tick here, so approximate phase as low (safe)
            return min(1.0, 0.25)
        except Exception:
            return 0.25
```
"""


if __name__ == "__main__":
    api_key = os.getenv("GEMINI_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")


    #llm1 = Gemini_LLM(api_key, "gemini-2.0-flash")
    
    llm1 = OpenAI_LLM(openai_api_key, "gpt-5-mini", temperature=1.0)
    llm2 = Gemini_LLM(api_key, "gemini-3-flash-preview")
    #llm2 = OpenAI_LLM(openai_api_key, "gpt-5.4", temperature=1.0)
    #llm3 = OpenAI_LLM(openai_api_key, "gpt-5.2", temperature=1.0)
    #llm3 = Gemini_LLM(api_key, "gemini-2.5-pro")
    #llm4 = Gemini_LLM(api_key, "gemini-2.5-flash")
    

    mllm = Multi_LLM(llms=[llm1,llm2])

    es = LLaMEA(
        evaluate_individual,
        llm=mllm,
        n_parents=5,
        n_offspring=20,
        budget=300,
        diff_mode=False,
        experiment_name="planetwars-individual-multi100",
        role_prompt=role_prompt,
        task_prompt=task_prompt,
        example_prompt=example_prompt,
        mutation_prompts=feedback_prompts,
        evaluate_population=False,
        elitism=True,
        adaptive_prompt=False,
        max_workers=5,
        parent_selection = "random",
        #tournament_size=2
    )

    print(es.run())

#Todo in the end, run a large tournament with all the best agents per generation
# and save the results to a file.