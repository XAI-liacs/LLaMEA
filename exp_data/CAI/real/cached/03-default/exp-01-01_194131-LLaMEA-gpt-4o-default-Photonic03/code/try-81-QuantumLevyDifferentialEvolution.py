import numpy as np

class QuantumLevyDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, 5 * dim)
        self.positions = np.random.rand(self.population_size, dim)
        self.trial_positions = np.zeros((self.population_size, dim))
        self.best_position = None
        self.best_score = float('inf')
        self.evaluations = 0
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability

    def levy_flight(self, scale=0.01):
        u = np.random.normal(0, 1, self.dim) * scale
        v = np.random.normal(0, 1, self.dim)
        step = u / (np.abs(v) ** (1 / 3))
        return step

    def quantum_state(self):
        alpha = np.random.rand(self.dim)
        beta = np.sqrt(1 - alpha**2)
        return alpha * self.best_position + beta * np.random.rand(self.dim)

    def _mutate_and_crossover(self, idx, func):
        indices = np.arange(self.population_size)
        np.random.shuffle(indices)
        a, b, c = self.positions[indices[:3]]
        
        mutant = np.clip(a + self.F * (b - c), func.bounds.lb, func.bounds.ub)
        crossover_mask = np.random.rand(self.dim) < self.CR
        self.trial_positions[idx] = np.where(crossover_mask, mutant, self.positions[idx])

        if np.random.rand() < 0.5:
            self.trial_positions[idx] += self.levy_flight()

        trial_score = func(self.trial_positions[idx])
        if trial_score < self.best_score:
            self.best_score = trial_score
            self.best_position = self.trial_positions[idx]

        return trial_score

    def __call__(self, func):
        self.positions = func.bounds.lb + (func.bounds.ub - func.bounds.lb) * np.random.rand(self.population_size, self.dim)
        for i in range(self.population_size):
            score = func(self.positions[i])
            if score < self.best_score:
                self.best_score = score
                self.best_position = self.positions[i]
            self.evaluations += 1
            if self.evaluations >= self.budget:
                return self.best_position

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                trial_score = self._mutate_and_crossover(i, func)
                if trial_score < func(self.positions[i]):
                    self.positions[i] = self.trial_positions[i]
                self.evaluations += 1
                if self.evaluations >= self.budget:
                    break

        return self.best_position