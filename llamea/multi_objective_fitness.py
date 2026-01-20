class Fitness:
    """
    A class for multi_objective fitness management.
    Meant for easy comparison, between fitness value.
    `Note`: Do NOT use sort on this value, sorting makes certain assumptions that cannot be guaranteed by multi-objective fitnesses.
    `Usage`: Init with a `Dictionary<String, Float>` type multi-objective fitness. Comparisons are not associated with dominance:
        * `a < b` : `a` strictly dominates `b` in mininimsation task, and visa-versa in maximisation task.
        * `a > b` : `a` strictly dominates `b` in maximisation task, and visa-versa in minimisation task.
        * `a ≤ b` : `a` either dominates or belong to same pareto-front for minimisation task for `b`.
        * `a ≥ b` : `a` either dominates or belong to same pareto-front for maximisation task for `b`.
        * `a == b` : `a` and `b` have exact same fitness for all .
    """
    def __init__(self, value: dict[str, float] | None = None):
        if value is None:
            self._fitness: dict[str, float] = {}
        else:
            self._fitness = value.copy()

    def keys(self):
        return self._fitness.keys()

    def __getitem__(self, key):
        return self._fitness.get(key, float("nan"))
    
    def __setitem__(self, key: str, value: float):
        self._fitness[key] = value

    def _dominates(self, other: "Fitness") -> tuple[bool, bool]:
        better_or_equal = all(self[k] <= other[k] for k in self.keys())
        strictly_better = any(self[k] < other[k] for k in self.keys())
        return better_or_equal, strictly_better

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Fitness):
            return False
        return self._fitness == other._fitness

    def __lt__(self, other: "Fitness") -> bool:
        if not isinstance(other, Fitness):
            return NotImplemented
        be, sb = self._dominates(other)
        return be and sb

    def __gt__(self, other: "Fitness") -> bool:
        if not isinstance(other, Fitness):
            return NotImplemented
        be, sb = other._dominates(self)
        return be and sb

    def __le__(self, other: "Fitness") -> bool:
        if not isinstance(other, Fitness):
            return NotImplemented
        be, sb = self._dominates(other)
        return be or sb

    def __ge__(self, other: "Fitness") -> bool:
        if not isinstance(other, Fitness):
            return NotImplemented
        be, sb = other._dominates(self)
        return be or sb
    
    def to_vector(self) -> list[float]:
        vector = []
        for objective in sorted(self.keys()):
            vector.append(self[objective])
        return vector

    def to_dict(self) -> dict[str, float]:
        return self._fitness
    
    @classmethod
    def from_dict(cls, data: dict[str, float]) -> "Fitness":
        """Construct Fitness from a dict (e.g., after json.loads)."""
        return cls(value=data)

    def __repr__(self) -> str:
        repr_str = ""
        for key, value in self._fitness.items():
            repr_str += f"{key} : {value}, "
        repr_str = repr_str[:-2]
        return repr_str
