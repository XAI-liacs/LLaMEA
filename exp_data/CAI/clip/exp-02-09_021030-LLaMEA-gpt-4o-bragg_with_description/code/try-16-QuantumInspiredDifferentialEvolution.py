import numpy as np

class QuantumInspiredDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.amplitude_factor = 0.5

    def initialize_population(self, bounds):
        self.population = np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim))
        self.fitness = np.array([float('inf')] * self.population_size)
        self.best_individual = None

    def __call__(self, func):
        bounds = func.bounds
        self.initialize_population(bounds)
        evaluations = 0
        best_score = float('inf')

        while evaluations < self.budget:
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = self.population[indices]
                mutant = np.clip(a + self.mutation_factor * (b - c), bounds.lb, bounds.ub)
                trial = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant, self.population[i])

                # Quantum-inspired probabilistic amplitude adjustment
                quantum_adjustment = (2 * np.random.rand(self.dim) - 1) * self.amplitude_factor
                trial = np.clip(trial + quantum_adjustment, bounds.lb, bounds.ub)

                trial_score = func(trial)
                evaluations += 1

                if trial_score < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_score
                    if trial_score < best_score:
                        best_score = trial_score
                        self.best_individual = trial

        return self.best_individual