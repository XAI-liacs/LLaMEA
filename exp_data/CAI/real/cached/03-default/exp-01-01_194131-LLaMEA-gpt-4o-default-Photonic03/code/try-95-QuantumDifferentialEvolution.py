import numpy as np

class QuantumDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, 5 * dim)
        self.positions = np.random.rand(self.population_size, dim)
        self.scores = np.full(self.population_size, float('inf'))
        self.gbest = None
        self.gbest_score = float('inf')
        self.evaluations = 0
        self.mutation_factor = 0.5
        self.crossover_rate = 0.7

    def _quantum_update(self, idx, func):
        a, b, c = np.random.choice(self.population_size, 3, replace=False)
        mutant = self.positions[a] + self.mutation_factor * (self.positions[b] - self.positions[c])
        mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

        trial = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant, self.positions[idx])
        trial_score = func(trial)

        if trial_score < self.scores[idx]:
            self.positions[idx] = trial
            self.scores[idx] = trial_score

        if trial_score < self.gbest_score:
            self.gbest = trial
            self.gbest_score = trial_score

        self.evaluations += 1

    def _dynamic_population(self):
        if self.evaluations % (self.budget // 4) == 0:
            self.mutation_factor = np.random.uniform(0.4, 0.9)
            self.crossover_rate = np.random.uniform(0.5, 0.9)

    def __call__(self, func):
        self.positions = func.bounds.lb + (func.bounds.ub - func.bounds.lb) * np.random.rand(self.population_size, self.dim)
        for i in range(self.population_size):
            score = func(self.positions[i])
            self.scores[i] = score
            if score < self.gbest_score:
                self.gbest = self.positions[i]
                self.gbest_score = score
            self.evaluations += 1
            if self.evaluations >= self.budget:
                return self.gbest

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                self._quantum_update(i, func)
                if self.evaluations >= self.budget:
                    break
            self._dynamic_population()

        return self.gbest