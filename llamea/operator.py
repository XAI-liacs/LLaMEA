"""
A operator class for managing mutation / crossover prompts.
"""

import uuid
from dataclasses import dataclass, field


@dataclass
class Operator:
    prompt: str
    name: str = "mutation"
    weight: float = field(default=0.5, compare=False)
    number_of_parents: int = 1

    def __post_init__(self):
        self.id = uuid.uuid4().hex
        self.rewards = 0.0

    def __repr__(self) -> str:
        return self.prompt or ""

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, value: 'Operator') -> bool:
        return self.id == value.id
