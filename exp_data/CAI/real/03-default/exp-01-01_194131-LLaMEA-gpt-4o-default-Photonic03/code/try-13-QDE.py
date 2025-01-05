import numpy as np

class QDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, 5 * dim)
        self.positions = np.random.rand(self.population_size, dim)
        self.best_global_position = None
        self.best_global_score = float('inf')
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover rate
        self.evaluations = 0

    def _quantum_superposition(self):
        """Generate a new solution based on quantum superposition principle."""
        return np.random.uniform(0, 1, self.dim)

    def _mutation(self, idx, func):
        """Perform differential mutation and crossover."""
        indices = list(range(self.population_size))
        indices.remove(idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        
        mutant = self.positions[a] + self.F * (self.positions[b] - self.positions[c])
        trial = np.where(np.random.rand(self.dim) < self.CR, mutant, self.positions[idx])
        trial = np.clip(trial, func.bounds.lb, func.bounds.ub)
        
        trial_score = func(trial)
        if trial_score < self.best_global_score:
            self.best_global_position = trial
            self.best_global_score = trial_score
        
        if trial_score < func(self.positions[idx]):
            self.positions[idx] = trial

        self.evaluations += 1

    def __call__(self, func):
        # Initialize with quantum superposition
        self.positions = func.bounds.lb + (func.bounds.ub - func.bounds.lb) * np.array([self._quantum_superposition() for _ in range(self.population_size)])
        for i in range(self.population_size):
            score = func(self.positions[i])
            if score < self.best_global_score:
                self.best_global_position = self.positions[i]
                self.best_global_score = score
            self.evaluations += 1
            if self.evaluations >= self.budget:
                return self.best_global_position

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                self._mutation(i, func)
                if self.evaluations >= self.budget:
                    break

        return self.best_global_position