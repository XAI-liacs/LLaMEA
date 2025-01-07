import numpy as np

class AdaptiveQuantumInspiredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.mutation_rate = 0.1
        self.mutation_strength = 0.1
        self.history = []
        self.qubit_population = np.ones((self.population_size, self.dim, 2)) / np.sqrt(2)  # Initialize in superposition

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        evaluations = 0
        best_solution = None
        best_fitness = np.inf

        while evaluations < self.budget:
            # Measure quantum population to get candidate solutions
            pop = np.array([np.random.choice([lb[i], ub[i]], p=qubit[:, 0]**2) for qubit in self.qubit_population])

            # Evaluate current population
            fitness = np.array([func(x) for x in pop])
            evaluations += self.population_size

            # Update best solution
            if best_solution is None or np.min(fitness) < best_fitness:
                best_fitness = np.min(fitness)
                best_solution = pop[np.argmin(fitness)]

            # Calculate probabilities based on fitness improvement
            avg_fitness = np.mean(fitness)
            improvement = np.maximum(0, avg_fitness - best_fitness)
            probability_update = improvement / (avg_fitness + 1e-8)

            # Adaptive mutation based on improvement
            self.mutation_rate = max(0.01, self.mutation_rate * (1 - probability_update))
            self.mutation_strength = np.clip(self.mutation_strength * (1 + probability_update), 0.01, 1.0)

            # Quantum Rotation based on success probability
            for i in range(self.population_size):
                angle_update = np.random.normal(0, self.mutation_strength, self.dim)
                for j in range(self.dim):
                    rotation_matrix = np.array([[np.cos(angle_update[j]), -np.sin(angle_update[j])],
                                                [np.sin(angle_update[j]), np.cos(angle_update[j])]])
                    self.qubit_population[i, j] = rotation_matrix @ self.qubit_population[i, j]

            self.history.append(best_solution)

        return best_solution