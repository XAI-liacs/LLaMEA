import numpy as np

class QuantumInspiredEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.alpha = 0.5  # Coefficient for quantum rotation
        self.beta = 0.9   # Coefficient for quantum mutation
        self.history = []

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        quantum_population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(x) for x in quantum_population])
        best_global = quantum_population[np.argmin(fitness)]
        evaluations = self.population_size

        while evaluations < self.budget:
            # Quantum rotation step
            rotation_matrix = self.alpha * (best_global - quantum_population)
            quantum_population += rotation_matrix
            quantum_population = np.clip(quantum_population, lb, ub)

            # Quantum mutation step
            quantum_population += self.beta * np.random.uniform(-1, 1, quantum_population.shape)

            # Ensure population stays within bounds
            quantum_population = np.clip(quantum_population, lb, ub)

            # Evaluate the new population
            fitness = np.array([func(x) for x in quantum_population])
            evaluations += self.population_size

            # Update the global best
            current_best_index = np.argmin(fitness)
            if fitness[current_best_index] < func(best_global):
                best_global = quantum_population[current_best_index]

            # Save the history of best solutions
            self.history.append(best_global)

        return best_global