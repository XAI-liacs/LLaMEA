import numpy as np

class QHDS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, budget // 2)
        self.positions = None
        self.best_position = None
        self.best_value = np.inf
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.randomize_rate = 0.3

    def initialize_population(self, lb, ub):
        self.positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.best_position = None
        self.best_value = np.inf
        self.bounds = (lb, ub)

    def quantum_harmonic_update(self, position, best_position):
        alpha = np.random.standard_normal(self.dim)
        gamma = np.random.normal(0, 0.1, self.dim)
        new_position = position + alpha * (best_position - position) + gamma
        lb, ub = self.bounds
        return np.clip(new_position, lb, ub)

    def differential_mutation(self, idx, population):
        candidates = list(range(self.population_size))
        candidates.remove(idx)
        a, b, c = np.random.choice(candidates, 3, replace=False)
        mutant_vector = population[a] + self.mutation_factor * (population[b] - population[c])
        lb, ub = self.bounds
        return np.clip(mutant_vector, lb, ub)

    def crossover(self, target, mutant):
        crossover_vector = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant, target)
        return crossover_vector

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                mutant_vector = self.differential_mutation(i, self.positions)
                trial_vector = self.crossover(self.positions[i], mutant_vector)
                
                if np.random.rand() < self.randomize_rate:
                    trial_vector = self.quantum_harmonic_update(trial_vector, self.best_position if self.best_position is not None else np.mean(self.positions, axis=0))
                
                trial_value = func(trial_vector)
                evaluations += 1

                if trial_value < func(self.positions[i]):
                    self.positions[i] = trial_vector

                if trial_value < self.best_value:
                    self.best_value = trial_value
                    self.best_position = trial_vector

        return self.best_position, self.best_value