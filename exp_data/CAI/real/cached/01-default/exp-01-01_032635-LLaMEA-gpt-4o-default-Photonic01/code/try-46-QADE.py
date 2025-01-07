import numpy as np

class QADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, budget)
        self.mutation_factor = 0.5
        self.crossover_probability = 0.9
        self.positions = None
        self.bounds = None

    def initialize_population(self, lb, ub):
        self.positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.bounds = (lb, ub)

    def quantum_mutation(self, target_idx):
        indices = list(range(self.population_size))
        indices.remove(target_idx)
        selected = np.random.choice(indices, 3, replace=False)
        a, b, c = self.positions[selected]
        mutation_vector = a + self.mutation_factor * (b - c)
        beta = np.random.normal(0, 1, self.dim)
        delta = np.random.normal(0, 1, self.dim)
        quantum_mutation_vector = mutation_vector + beta * (self.positions[target_idx] - mutation_vector) + delta * 0.1
        lb, ub = self.bounds
        return np.clip(quantum_mutation_vector, lb, ub)

    def crossover(self, target_vector, donor_vector):
        crossover_mask = np.random.rand(self.dim) < self.crossover_probability
        trial_vector = np.where(crossover_mask, donor_vector, target_vector)
        lb, ub = self.bounds
        return np.clip(trial_vector, lb, ub)

    def adapt_parameters(self, iteration, max_iterations):
        self.mutation_factor = 0.5 + 0.5 * (iteration / max_iterations)
        self.crossover_probability = 0.9 - 0.4 * (iteration / max_iterations)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        evaluations = 0
        best_position = None
        best_value = np.inf

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                donor_vector = self.quantum_mutation(i)
                trial_vector = self.crossover(self.positions[i], donor_vector)
                trial_value = func(trial_vector)
                evaluations += 1

                if trial_value < func(self.positions[i]):
                    self.positions[i] = trial_vector

                if trial_value < best_value:
                    best_value = trial_value
                    best_position = trial_vector

            self.adapt_parameters(evaluations, self.budget)

        return best_position, best_value