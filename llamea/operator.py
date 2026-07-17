"""
A operator class for managing mutation / crossover prompts.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional



from llamea.multi_objective_fitness import Fitness


@dataclass
class Operator:
    prompt: str
    name: str='mutation'
    weight: float = field(default=0.5, compare=False)
    number_of_parents: int=1

    update_weight_function : Optional[Callable[[list[float | Fitness], float | Fitness, bool], list[int]]] = field(default=None, compare=False)

    def __post_init__(self):
        self._fitness_contribution = 0.0
        self._use_count = 0

        def update_function(
            parent_fitness: [float | Fitness],
            offspring_fitness: float | Fitness,
            minimisation: bool
        ):
            def get_fitness_value(f):
                return f.to_vector() if isinstance(f, Fitness) else [f]

            offspring_value = get_fitness_value(offspring_fitness)
            parent_values = [get_fitness_value(f) for f in parent_fitness]

            dominance_vector = []

            if not any([np.isfinite(value) for value in offspring_value]):
                dominance_vector = [-1] * self.number_of_parents
            else:
                for fitness_value in parent_values:
                    if minimisation:
                        element = (-1) ** int(fitness_value < offspring_value)
                    else:
                        element = (-1) ** int(fitness_value > offspring_value)

                    dominance_vector.append(element)

            self._fitness_contribution += sum(dominance_vector) / self.number_of_parents
            self._use_count += 1
            self.weight = 1 / (1 + np.e ** (-self._fitness_contribution / self._use_count))

            return dominance_vector

        if self.update_weight_function is None:
            self.update_weight_function = update_function


    def __repr__(self) -> str:
        return self.prompt or ""

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __hash__(self):
        return hash((self.promt, self.name, self.number_of_parents))
