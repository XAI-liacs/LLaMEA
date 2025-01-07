import numpy as np

class QDELAF:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, 5 * dim)
        self.positions = np.random.rand(self.population_size, dim)
        self.scores = np.full(self.population_size, float('inf'))
        self.best_position = None
        self.best_score = float('inf')
        self.scaling_factor = 0.5
        self.crossover_rate = 0.7
        self.beta = 0.2
        self.evaluations = 0

    def levy_flight(self, scale=0.01):
        u = np.random.normal(0, 1, self.dim) * scale
        v = np.random.normal(0, 1, self.dim)
        step = u / (np.abs(v) ** (1 / 3))
        return step

    def _mutate(self, idx):
        indices = np.arange(self.population_size)
        indices = indices[indices != idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = self.positions[a] + self.scaling_factor * (self.positions[b] - self.positions[c])
        return np.clip(mutant, 0, 1)

    def _crossover(self, parent, mutant):
        crossover = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant, parent)
        return crossover

    def _select(self, idx, trial_pos, trial_score):
        if trial_score < self.scores[idx]:
            self.positions[idx] = trial_pos
            self.scores[idx] = trial_score
            if trial_score < self.best_score:
                self.best_position = trial_pos
                self.best_score = trial_score

    def _adaptive_beta(self):
        if self.evaluations % (self.budget // 10) == 0:
            self.beta = min(1.0, self.beta + 0.05)

    def __call__(self, func):
        self.positions = func.bounds.lb + (func.bounds.ub - func.bounds.lb) * np.random.rand(self.population_size, self.dim)
        for i in range(self.population_size):
            self.scores[i] = func(self.positions[i])
            if self.scores[i] < self.best_score:
                self.best_position = self.positions[i]
                self.best_score = self.scores[i]
            self.evaluations += 1
            if self.evaluations >= self.budget:
                return self.best_position

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                mutant = self._mutate(i)
                trial = self._crossover(self.positions[i], mutant)
                
                if np.random.rand() < self.beta:
                    trial += self.levy_flight()

                trial = np.clip(trial, func.bounds.lb, func.bounds.ub)
                trial_score = func(trial)

                self._select(i, trial, trial_score)
                self.evaluations += 1
                if self.evaluations >= self.budget:
                    break

            self._adaptive_beta()

        return self.best_position