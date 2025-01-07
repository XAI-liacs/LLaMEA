import numpy as np

class DEQCC:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, 5 * dim)
        self.positions = np.random.rand(self.population_size, dim)
        self.best_position = None
        self.best_score = float('inf')
        self.evaluations = 0
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.dynamic_groups = max(2, dim // 5)

    def _quantum_mutation(self, target_idx, func):
        idxs = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = np.random.choice(idxs, 3, replace=False)
        mutant = self.positions[a] + self.F * (self.positions[b] - self.positions[c])
        crossover = np.random.rand(self.dim) < self.CR
        crossover[np.random.randint(0, self.dim)] = True
        trial = np.where(crossover, mutant, self.positions[target_idx])
        trial = np.clip(trial, func.bounds.lb, func.bounds.ub)
        return trial

    def _evaluate_and_update(self, pos, func, idx):
        score = func(pos)
        if score < self.best_score:
            self.best_score = score
            self.best_position = pos
        if score < func(self.positions[idx]):
            self.positions[idx] = pos
        self.evaluations += 1

    def _dynamic_grouping(self):
        group_size = max(1, self.dim // self.dynamic_groups)
        groups = [np.random.permutation(self.dim)[:group_size] for _ in range(self.dynamic_groups)]
        return groups

    def __call__(self, func):
        self.positions = func.bounds.lb + (func.bounds.ub - func.bounds.lb) * np.random.rand(self.population_size, self.dim)
        for idx in range(self.population_size):
            self._evaluate_and_update(self.positions[idx], func, idx)
            if self.evaluations >= self.budget:
                return self.best_position

        while self.evaluations < self.budget:
            groups = self._dynamic_grouping()
            for group in groups:
                for idx in range(self.population_size):
                    trial = self._quantum_mutation(idx, func)
                    trial[group] = self.positions[idx][group]  # Cooperative Coevolution
                    self._evaluate_and_update(trial, func, idx)
                    if self.evaluations >= self.budget:
                        return self.best_position

        return self.best_position