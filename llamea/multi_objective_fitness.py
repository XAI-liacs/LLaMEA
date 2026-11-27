import math

_INTERFACE = ("keys", "to_vector", "to_dict")


def is_multi_objective_fitness(value) -> bool:
    """True for a multi-objective fitness, of this or another class.

    Any object with ``keys()``, ``to_vector()`` and ``to_dict()`` counts, so
    LLaMEA works with evaluators that use their own fitness class (e.g.
    BLADE's ``iohblade.fitness.Fitness``).
    """
    return all(callable(getattr(value, name, None)) for name in _INTERFACE)


def _values(other) -> dict | None:
    """The objective values of a multi-objective fitness, or None."""
    if isinstance(other, Fitness):
        return other._fitness
    if is_multi_objective_fitness(other):
        return dict(other.to_dict())
    return None


class Fitness:
    """Multi-objective fitness: one named value per objective.

    Create it from a ``dict[str, float]``, e.g.
    ``Fitness({"Distance": 12.3, "Fuel": 4.5})``. The comparison operators
    express Pareto dominance:

    * ``a < b``: ``a`` strictly dominates ``b`` when minimising (and ``b``
      dominates ``a`` when maximising).
    * ``a > b``: ``a`` strictly dominates ``b`` when maximising (and ``b``
      dominates ``a`` when minimising).
    * ``a <= b``: ``a`` dominates ``b`` or both are on the same Pareto front
      (minimisation).
    * ``a >= b``: ``a`` dominates ``b`` or both are on the same Pareto front
      (maximisation).
    * ``a == b``: all objective values are equal.

    Note:
        Do not sort lists of ``Fitness`` objects: dominance is only a partial
        order, so the result of a sort is not meaningful.
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

    @staticmethod
    def _dominance(a: dict, b: dict) -> tuple[bool, bool]:
        """(``a`` better or equal on all objectives, strictly better on one)."""
        nan = float("nan")
        better_or_equal = all(a.get(k, nan) <= b.get(k, nan) for k in a)
        strictly_better = any(a.get(k, nan) < b.get(k, nan) for k in a)
        return better_or_equal, strictly_better

    def _dominates(self, other) -> tuple[bool, bool]:
        return self._dominance(self._fitness, _values(other) or {})

    # Comparisons accept any multi-objective fitness (see
    # is_multi_objective_fitness), not only this class.

    def __eq__(self, other: object) -> bool:
        values = _values(other)
        if values is None:
            return False
        return self._fitness == values

    __hash__ = None  # mutable; equality is by value

    def __lt__(self, other) -> bool:
        values = _values(other)
        if values is None:
            return False
        be, sb = self._dominance(self._fitness, values)
        return be and sb

    def __gt__(self, other) -> bool:
        values = _values(other)
        if values is None:
            return False
        be, sb = self._dominance(values, self._fitness)
        return be and sb

    def __le__(self, other) -> bool:
        values = _values(other)
        if values is None:
            return False
        be, sb = self._dominance(self._fitness, values)
        return be or sb

    def __ge__(self, other) -> bool:
        values = _values(other)
        if values is None:
            return False
        be, sb = self._dominance(values, self._fitness)
        return be or sb

    def to_vector(self) -> list[float]:
        vector = []
        for objective in sorted(self.keys()):
            vector.append(self[objective])
        return vector

    def to_dict(self) -> dict[str, float]:
        return dict(self._fitness)

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> "Fitness":
        """Construct Fitness from a dict (e.g., after json.loads)."""
        return cls(value=data)

    def __float__(self) -> float:
        return (
            math.nan
            if any(math.isnan(value) for value in self._fitness.values())
            else sum(self.to_vector()) / len(self._fitness)
        )

    def __repr__(self) -> str:
        repr_str = ""
        for key, value in self._fitness.items():
            repr_str += f"{key} : {value}, "
        repr_str = repr_str[:-2]
        return repr_str
