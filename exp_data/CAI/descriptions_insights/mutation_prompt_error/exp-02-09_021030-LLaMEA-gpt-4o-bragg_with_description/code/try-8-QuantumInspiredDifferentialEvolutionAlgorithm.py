import numpy as np

class QuantumInspiredDifferentialEvolutionAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.mutation_factor = 0.8
        self.crossover_prob = 0.9
        self.quantum_prob = 0.1  # Probability of using quantum-inspired principles

    def initialize_population(self, bounds):
        self.population = np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim))
        self.best_global_position = self.population[np.random.randint(self.population_size)]

    def __call__(self, func):
        bounds = func.bounds
        self.initialize_population(bounds)
        evaluations = 0
        best_score = float('-inf')

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Mutate
                indices = np.random.choice(self.population_size, 3, replace=False)
                donor_vector = self.population[indices[0]] + self.mutation_factor * (self.population[indices[1]] - self.population[indices[2]])

                # Crossover
                trial_vector = np.copy(self.population[i])
                for j in range(self.dim):
                    if np.random.rand() < self.crossover_prob:
                        trial_vector[j] = donor_vector[j]

                # Quantum-inspired perturbation
                if np.random.rand() < self.quantum_prob:
                    trial_vector = self.quantum_perturbation(trial_vector, bounds)

                # Selection
                trial_vector = np.clip(trial_vector, bounds.lb, bounds.ub)
                trial_score = func(trial_vector)
                evaluations += 1

                if trial_score > func(self.population[i]):
                    self.population[i] = trial_vector

                if trial_score > best_score:
                    best_score = trial_score
                    self.best_global_position = trial_vector

        return self.best_global_position

    def quantum_perturbation(self, vector, bounds):
        # Use quantum superposition principle for perturbation
        q_vector = np.random.uniform(-1, 1, self.dim)
        amplitude = np.linalg.norm(vector - self.best_global_position)
        q_vector = vector + amplitude * q_vector
        return np.clip(q_vector, bounds.lb, bounds.ub)