"""
A operator class for managing mutation / crossover prompts.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Operator:
    name: Optional[str]
    prompt: Optional[str]
    weight: Optional[float]
    number_of_parents: Optional[int]

    def __post_init__(self):
        if self.name is None:
            self.name = "mutation"
        if self.weight is None:
            self.weight = 1.0
        if self.number_of_parents is None:
            self.number_of_parents = 1

    def __repr__(self) -> str:
        return self.prompt or ""
