import numpy as np

class QuantumInspiredDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 40
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.quantum_factor = 0.05

    def initialize(self, bounds):
        self.population = np.random.uniform(bounds.lb, bounds.ub, (self.pop_size, self.dim))
        self.scores = np.full(self.pop_size, float('inf'))
        self.best_idx = 0
        self.best_score = float('inf')

    def quantum_update(self, candidate, bounds):
        quantum_shift = self.quantum_factor * (2 * np.random.random(self.dim) - 1)
        new_candidate = candidate + quantum_shift
        return np.clip(new_candidate, bounds.lb, bounds.ub)

    def mutate_and_crossover(self, target_idx, bounds):
        indices = [idx for idx in range(self.pop_size) if idx != target_idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = self.population[a] + self.mutation_factor * (self.population[b] - self.population[c])
        mutant = self.quantum_update(mutant, bounds)
        trial = np.copy(self.population[target_idx])
        crossover_mask = np.random.rand(self.dim) < self.crossover_rate
        trial[crossover_mask] = mutant[crossover_mask]
        return np.clip(trial, bounds.lb, bounds.ub)

    def __call__(self, func):
        bounds = func.bounds
        self.initialize(bounds)

        evaluations = 0
        while evaluations < self.budget:
            for i in range(self.pop_size):
                trial = self.mutate_and_crossover(i, bounds)
                trial_score = func(trial)
                evaluations += 1

                if trial_score < self.scores[i]:
                    self.population[i] = trial
                    self.scores[i] = trial_score
                    if trial_score < self.best_score:
                        self.best_score = trial_score
                        self.best_idx = i

        best_position = self.population[self.best_idx]
        return best_position, self.best_score