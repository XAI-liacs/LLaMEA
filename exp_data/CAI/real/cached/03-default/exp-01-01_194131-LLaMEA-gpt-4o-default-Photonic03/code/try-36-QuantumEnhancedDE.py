import numpy as np

class QuantumEnhancedDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, 5 * dim)
        self.positions = np.random.rand(self.population_size, dim)
        self.best_pos = None
        self.best_score = float('inf')
        self.evaluations = 0
        self.mutation_factor = 0.5
        self.crossover_rate = 0.9

    def _quantum_initialize(self, bounds):
        theta = np.random.rand(self.population_size, self.dim) * 2 * np.pi
        r = np.random.rand(self.population_size, self.dim)
        self.positions = bounds.lb + (0.5 * np.abs(np.cos(theta)) * (bounds.ub - bounds.lb))

    def _adaptive_mutation(self, f_idx, bounds):
        a, b, c = np.random.choice([i for i in range(self.population_size) if i != f_idx], 3, replace=False)
        mutant_vector = self.positions[a] + self.mutation_factor * (self.positions[b] - self.positions[c])
        mutant_vector = np.clip(mutant_vector, bounds.lb, bounds.ub)
        return mutant_vector

    def _crossover(self, target_vector, mutant_vector):
        crossover_mask = np.random.rand(self.dim) < self.crossover_rate
        trial_vector = np.where(crossover_mask, mutant_vector, target_vector)
        return trial_vector

    def __call__(self, func):
        self._quantum_initialize(func.bounds)
        scores = np.full(self.population_size, float('inf'))

        for i in range(self.population_size):
            score = func(self.positions[i])
            scores[i] = score
            if score < self.best_score:
                self.best_score = score
                self.best_pos = self.positions[i]
            self.evaluations += 1
            if self.evaluations >= self.budget:
                return self.best_pos

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                mutant_vector = self._adaptive_mutation(i, func.bounds)
                trial_vector = self._crossover(self.positions[i], mutant_vector)
                trial_score = func(trial_vector)

                if trial_score < scores[i]:
                    self.positions[i] = trial_vector
                    scores[i] = trial_score
                    if trial_score < self.best_score:
                        self.best_score = trial_score
                        self.best_pos = trial_vector

                self.evaluations += 1
                if self.evaluations >= self.budget:
                    break

        return self.best_pos