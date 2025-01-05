import numpy as np

class QADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, budget)
        self.positions = None
        self.fitness = None
        self.best_position = None
        self.best_value = np.inf
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.adapt_rate = 0.1
        self.bounds = None

    def initialize_population(self, lb, ub):
        self.positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.bounds = (lb, ub)

    def quantum_position_update(self, position):
        alpha = np.random.normal(0, 1, self.dim)
        beta = np.random.normal(0, 1, self.dim)
        new_position = position + alpha * (self.best_position - position) + beta * 0.1
        lb, ub = self.bounds
        return np.clip(new_position, lb, ub)

    def differential_mutation(self, idx):
        candidates = list(range(self.population_size))
        candidates.remove(idx)
        a, b, c = np.random.choice(candidates, 3, replace=False)
        donor_vector = self.positions[a] + self.F * (self.positions[b] - self.positions[c])
        return np.clip(donor_vector, *self.bounds)

    def crossover(self, target, donor):
        crossover_mask = np.random.rand(self.dim) < self.CR
        trial_vector = np.where(crossover_mask, donor, target)
        return trial_vector

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                target = self.positions[i]
                donor = self.differential_mutation(i)
                trial = self.crossover(target, donor)

                if np.random.rand() < self.adapt_rate:
                    trial = self.quantum_position_update(trial)

                trial_value = func(trial)
                evaluations += 1

                if trial_value < self.fitness[i]:
                    self.positions[i] = trial
                    self.fitness[i] = trial_value

                if trial_value < self.best_value:
                    self.best_value = trial_value
                    self.best_position = trial

        return self.best_position, self.best_value