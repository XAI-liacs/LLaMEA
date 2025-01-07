import numpy as np

class QuantumEnhancedDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, 5 * dim)
        self.positions = np.random.rand(self.population_size, dim)
        self.pbest = self.positions.copy()
        self.gbest = None
        self.pbest_scores = np.full(self.population_size, float('inf'))
        self.gbest_score = float('inf')
        self.evaluations = 0
        self.F = 0.5  # Scaling factor for differential mutation
        self.CR = 0.9  # Crossover probability
        self.levy_scale = 0.01

    def levy_flight(self):
        u = np.random.normal(0, 1, self.dim) * self.levy_scale
        v = np.random.normal(0, 1, self.dim)
        step = u / (np.abs(v) ** (1 / 3))
        return step

    def _mutate(self, idx):
        indices = np.random.choice(self.population_size, 3, replace=False)
        while idx in indices:
            indices = np.random.choice(self.population_size, 3, replace=False)
        x1, x2, x3 = self.positions[indices]
        mutant = x1 + self.F * (x2 - x3)
        return mutant

    def _crossover(self, idx, mutant):
        trial = np.where(np.random.rand(self.dim) < self.CR, mutant, self.positions[idx])
        if np.random.rand() < 0.2:  # Quantum-inspired mutation
            trial += self.levy_flight()
        return trial

    def _select(self, idx, trial, func):
        trial_score = func(trial)
        if trial_score < self.pbest_scores[idx]:
            self.pbest[idx] = trial
            self.pbest_scores[idx] = trial_score
        if trial_score < self.gbest_score:
            self.gbest = trial
            self.gbest_score = trial_score
        if trial_score < func(self.positions[idx]):
            self.positions[idx] = trial
        self.evaluations += 1

    def __call__(self, func):
        self.positions = func.bounds.lb + (func.bounds.ub - func.bounds.lb) * np.random.rand(self.population_size, self.dim)
        for i in range(self.population_size):
            score = func(self.positions[i])
            self.pbest[i] = self.positions[i]
            self.pbest_scores[i] = score
            if score < self.gbest_score:
                self.gbest = self.positions[i]
                self.gbest_score = score
            self.evaluations += 1
            if self.evaluations >= self.budget:
                return self.gbest

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                mutant = self._mutate(i)
                trial = self._crossover(i, mutant)
                trial = np.clip(trial, func.bounds.lb, func.bounds.ub)
                self._select(i, trial, func)
                if self.evaluations >= self.budget:
                    break

        return self.gbest