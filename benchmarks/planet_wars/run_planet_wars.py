import os

from benchmarks.planet_wars.evaluate import evaluate_tournament
from llamea import Gemini_LLM, LLaMEA

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
    "Either refine or redesign to improve the solution (and give it a distinct one-line description)."
]

example_prompt = 
""""An example of an AI agent, is as follows:

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
""""


def evaluate(solution, explogger=None):
    # Run a small tournament against built-in baselines
    evaluate_tournament(
        [solution], scenarios=["default"], game_dir=os.getenv("PLANET_WARS_DIR")
    )
    return solution


if __name__ == "__main__":
    api_key = os.getenv("GEMINI_API_KEY")
    ai_model = "gemini-2.0-flash"
    llm = Gemini_LLM(api_key, ai_model)

    es = LLaMEA(
        evaluate,
        llm=llm,
        n_parents=6,
        n_offspring=12,
        budget=100,
        experiment_name="planet_wars_openended",
        role_prompt=role_prompt,
        task_prompt=task_prompt,
        example_prompt=example_prompt,
        mutation_prompts=feedback_prompts,
        selection_strategy="tournament",
        tournament_size=4,
    )

    print(es.run())
