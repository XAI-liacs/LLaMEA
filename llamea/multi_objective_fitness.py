class Fitness:
    """
    A class for multi_objective fitness management.
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
        return not (self < other) and not (other < self)

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
        a = self < other
        b = self == other
        return a or b

    def __ge__(self, other: "Fitness") -> bool:
        if not isinstance(other, Fitness):
            return NotImplemented
        a = self > other
        b = self == other
        return a or b