import os

from benchmarks.planet_wars.evaluate import evaluate_tournament
from llamea import Gemini_LLM, LLaMEA

role_prompt = (
    "You are an expert game AI developer specialised in real-time strategy games."
)

# Task prompt instructs the LLM to implement a Planet Wars agent in Python
# using the provided interface.
task_prompt = """
Implement a Python agent for the Planet Wars RTS game. The agent should
inherit from `agents.planet_wars_agent.PlanetWarsPlayer` and implement the
`get_action` and `get_agent_type` methods. Focus on robust behaviour across
different game setups.
"""

feedback_prompts = [
    "Either refine or redesign to improve the solution (and give it a distinct one-line description)."
]


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
        mutation_prompts=feedback_prompts,
        selection_strategy="tournament",
        tournament_size=4,
    )

    print(es.run())
