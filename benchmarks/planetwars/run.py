import os
import json
from llamea import Gemini_LLM, OpenAI_LLM, LLaMEA
from pathlib import Path
import subprocess, json, os

PLANETWARS_DIR = "/home/neocortex/repos/planet-wars-rts-python"

GAME_DIR = Path(PLANETWARS_DIR) / "app" / "src" / "main" / "python"
VENV_PY = Path(PLANETWARS_DIR) / ".venv" / "bin" / "python"


def evaluate_tournament(solutions, logger=None, timeout=3600):
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
- numpy version 2.2.6 for numerical computations
- scipy version 1.15.3 for scientific computing
"""

# Task prompt that combines the detailed game description with the actual
# request for an agent implementation.
task_prompt = (
    game_info
    + "\nImplement a Python agent for the Planet Wars RTS game. The agent should"
    + " inherit from `agents.planet_wars_agent.PlanetWarsPlayer` and implement"
    + " the `get_action` and `get_agent_type` methods. Focus on robust"
    + " behaviour across different game setups."
)

feedback_prompts = [
    "Either refine or redesign to improve the solution (and give it a distinct one-line description and distinct name).",
    "Improve the solution by addressing its weaknesses and enhancing its performance. Give it a distinct one-line description and distinct name.",
    "Given the list of already implemented agents, generate a new agent that is completely different from the existing ones. Give it a distinct one-line description and distinct name.",
]

example_prompt = """"An example of an AI agent, is as follows:

```python
import random
from typing import Optional

from agents.planet_wars_agent import PlanetWarsPlayer
from core.game_state import GameState, Action, Player, GameParams
from core.game_state_factory import GameStateFactory


class GreedyHeuristicAgent(PlanetWarsPlayer):
    def get_action(self, game_state: GameState) -> Action:
        # Filter own planets that are not busy and have enough ships
        my_planets = [p for p in game_state.planets
                      if p.owner == self.player and p.transporter is None and p.n_ships > 10]
        if not my_planets:
            return Action.do_nothing()

        # Consider planets not owned by the player
        candidate_targets = [p for p in game_state.planets if p.owner != self.player]
        if not candidate_targets:
            return Action.do_nothing()

        # Pick source planet with the most ships
        source = max(my_planets, key=lambda p: p.n_ships)

        # Heuristic: prefer weak, nearby, fast-growing targets
        def target_score(target):
            distance = source.position.distance(target.position)
            ship_strength = target.n_ships if target.owner == Player.Neutral else target.n_ships * 1.5
            return ship_strength + distance - 2 * target.growth_rate

        target = min(candidate_targets, key=target_score)

        # Estimate whether the attack would succeed
        distance = source.position.distance(target.position)
        eta = distance / self.params.transporter_speed
        estimated_defense = target.n_ships + target.growth_rate * eta

        if source.n_ships <= estimated_defense:
            return Action.do_nothing()

        return Action(
            player_id=self.player,
            source_planet_id=source.id,
            destination_planet_id=target.id,
            num_ships=source.n_ships / 2
        )

    def get_agent_type(self) -> str:
        return "Greedy Heuristic Agent in Python"


# Example usage
if __name__ == "__main__":
    agent = GreedyHeuristicAgent()
    agent.prepare_to_play_as(Player.Player1, GameParams())
    game_state = GameStateFactory(GameParams()).create_game()
    action = agent.get_action(game_state)
    print(action)
```
"""

example_prompt = "An example of an AI agent is as follows:\n\n"+"""
```python
import random
from typing import Optional

from agents.planet_wars_agent import PlanetWarsPlayer
from core.game_state import GameState, Action, Player, GameParams
from core.game_state_factory import GameStateFactory


class GalacticArmada(PlanetWarsPlayer):
    def get_action(self, game_state: GameState) -> Action:
        my_planets = [p for p in game_state.planets
                      if p.owner == self.player and p.transporter is None and p.n_ships > 0]
        enemy_planets = [p for p in game_state.planets if p.owner == self.player.opponent()]
        neutral_planets = [p for p in game_state.planets if p.owner == Player.Neutral]
        all_planets = my_planets + enemy_planets + neutral_planets

        if not my_planets:
            return Action.do_nothing()

        # Prioritize attacking enemy planets if they exist, otherwise attack neutral planets
        targets = enemy_planets if enemy_planets else neutral_planets

        if not targets:
            return Action.do_nothing()

        # Sort my planets by ship count, descending
        my_planets = sorted(my_planets, key=lambda p: p.n_ships, reverse=True)

        # Find weakest friendly planet to defend
        weakest_planet = None
        if my_planets:
            weakest_planet = min(my_planets, key=lambda p: p.n_ships + p.growth_rate * 5) # 5 turns of growth

        # Consolidate forces on the strongest planet if close enough, but limit distance and ship count.
        strongest_planet = my_planets[0]
        for planet in my_planets[1:]:
            if planet.position.distance(strongest_planet.position) < 12 and planet.n_ships > 5 and strongest_planet.n_ships < 100:
                transfer_amount = int(min(planet.n_ships * 0.2, 40))  # Reduced transfer amount to conserve ships
                return Action(
                    player_id=self.player,
                    source_planet_id=planet.id,
                    destination_planet_id=strongest_planet.id,
                    num_ships=transfer_amount
                )

        # Dynamic target evaluation incorporating distance, ship strength and growth rate
        for source in my_planets:
            def target_score(target):
                distance = source.position.distance(target.position)
                ship_strength = target.n_ships
                if target.owner == Player.Neutral:
                    ship_strength *= 0.7  # Even less aggressive on neutrals
                else:
                    ship_strength *= 1.8 # Still aggressive on enemies, but slightly reduced
                growth_rate = target.growth_rate
                # Add a small random factor to break ties and explore diverse targets
                random_factor = random.uniform(-0.2, 0.2)
                return ship_strength + distance - 2.5 * growth_rate + random_factor# Prioritize nearby planets with high growth rate and low ship count.  Slightly reduced growth rate influence

            target = min(targets, key=target_score)

            # Enhanced ship allocation strategy: Send enough ships for a successful attack, plus some margin
            distance = source.position.distance(target.position)
            eta = distance / self.params.transporter_speed
            estimated_defense = target.n_ships + target.growth_rate * eta

            # Threat assessment: Increase the safety margin if the target is an enemy planet
            safety_margin = 1.2 if target.owner == self.player.opponent() else 1.05

            required_ships = int(estimated_defense * safety_margin)
            available_ships = source.n_ships
            #Conserve ships: Send a smaller percentage of ships, but more effectively.
            send_ships = min(available_ships, required_ships, int(available_ships * 0.7))

            if send_ships > 0:
                return Action(
                    player_id=self.player,
                    source_planet_id=source.id,
                    destination_planet_id=target.id,
                    num_ships=send_ships
                )

        # Improved defense action: Only defend if the weakest planet is under threat
        if weakest_planet is not None:
            # Check for nearby enemy planets that could attack the weakest planet
            nearby_enemies = [p for p in enemy_planets if p.position.distance(weakest_planet.position) < 20]

            # Consider enemies further away as well, scaled by distance
            far_enemies = [p for p in enemy_planets if 20 <= p.position.distance(weakest_planet.position) < 40]
            scaled_far_enemies = [(p, 0.3) for p in far_enemies] # Reduce the threat from far enemies more strongly

            all_threatening_enemies = nearby_enemies + [p[0] for p in scaled_far_enemies] # Only planets

            if all_threatening_enemies:
                # Assess the threat level: if the combined fleet size of nearby enemies is larger than the defense of the weakest planet, defend it.
                enemy_fleet_size = sum(p.n_ships for p in nearby_enemies) + sum(p.n_ships * 0.3 for p in far_enemies) #Weighted sum to take distance into account
                # Predict enemy arrival based on distance and transporter speed
                arrival_ticks = min(((weakest_planet.position.distance(enemy.position) / self.params.transporter_speed) for enemy in nearby_enemies), default=10)

                defense = weakest_planet.n_ships + weakest_planet.growth_rate * arrival_ticks #estimate growth until arrival

                if enemy_fleet_size > defense:
                    # Prioritize defense by checking if we can overwhelm the enemy attack from several planets at once
                    defending_planets = [p for p in my_planets if p.id != weakest_planet.id and p.n_ships > 10 and p.position.distance(weakest_planet.position) < 30]
                    total_available_ships = sum(min(p.n_ships * 0.4, 40) for p in defending_planets) # Reduced ship contribution, further reducing over-committing
                    if total_available_ships > (enemy_fleet_size - defense):
                        # Coordinate defense with multiple planets
                        for planet in defending_planets:
                            transfer_amount = int(min(planet.n_ships * 0.4, 40)) #Reduced ship contribution
                            if transfer_amount > 0:
                                action = Action(
                                    player_id=self.player,
                                    source_planet_id=planet.id,
                                    destination_planet_id=weakest_planet.id,
                                    num_ships=transfer_amount
                                )
                                return action
                    else:
                        # Transfer ships from the strongest planet to the weakest planet
                        if strongest_planet.id != weakest_planet.id and strongest_planet.n_ships > 20:
                            transfer_amount = int(min(strongest_planet.n_ships * 0.2, 50))  # Reduced transfer amount
                            action = Action(
                                player_id=self.player,
                                source_planet_id=strongest_planet.id,
                                destination_planet_id=weakest_planet.id,
                                num_ships=transfer_amount
                            )
                            return action

        return Action.do_nothing()

    def get_agent_type(self) -> str:
        return "Galactic Armada"
```
"""

if __name__ == "__main__":
    api_key = os.getenv("GEMINI_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    ai_model = "gemini-2.0-flash" #"o4-mini-2025-04-16" #"o3-mini-2025-01-31"
    llm = Gemini_LLM(api_key, ai_model)
    llm2 = OpenAI_LLM(openai_api_key, "gpt-5-2025-08-07", temperature=1.0)

    es = LLaMEA(
        evaluate_tournament,
        llm=llm2,
        n_parents=2,
        n_offspring=4,
        budget= 6,
        diff_mode=True,
        experiment_name="planetwars",
        role_prompt=role_prompt,
        task_prompt=task_prompt,
        example_prompt=example_prompt,
        mutation_prompts=feedback_prompts,
        evaluate_population=True,
        elitism=False,
        max_workers=2
    )

    print(es.run())

#Todo in the end, run a large tournament with all the best agents per generation
# and save the results to a file.