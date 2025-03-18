import numpy as np

class QuantumInspiredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, int(0.02 * budget))  # Dynamic population size
        self.mutation_rate = 0.1
        self.q = np.full((self.population_size, dim), 0.5)  # Quantum probabilities

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)

    def measure(self):
        """Measure operation in quantum computing to collapse states."""
        return np.where(np.random.rand(*self.q.shape) < self.q, 1, 0)

    def update_quantum_probabilities(self, best_solution):
        """Update quantum probabilities based on the best solution."""
        for i in range(self.population_size):
            fitness_factor = 1 + 0.05 * (self.fitness[i] - np.min(self.fitness))  # Adaptivity
            for j in range(self.dim):
                if self.population[i, j] == best_solution[j]:
                    self.q[i, j] = min(0.9, self.q[i, j] + 0.01 * fitness_factor)
                else:
                    self.q[i, j] = max(0.1, self.q[i, j] - 0.01 * fitness_factor)

    def evolutionary_process(self, lb, ub):
        """Perform evolutionary operations."""
        # Quantum-inspired offspring generation
        offspring = np.clip(np.random.normal(loc=self.q, scale=0.1), 0, 1)  # Gaussian-based measure
        
        # Map offspring to real values within bounds
        real_values = lb + (ub - lb) * (offspring / (self.q.shape[1] - 1))
        
        # Dynamic mutation rate based on fitness variance
        fitness_variance = np.var(self.fitness)
        dynamic_mutation_rate = self.mutation_rate + 0.1 * fitness_variance
        
        # Mutation
        mutation_mask = np.random.rand(*real_values.shape) < dynamic_mutation_rate
        real_values[mutation_mask] += np.random.normal(0, 0.1, real_values[mutation_mask].shape)
        np.clip(real_values, lb, ub, out=real_values)  # Ensure within bounds

        return real_values

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        eval_count = 0
        
        while eval_count < self.budget:
            new_population = self.evolutionary_process(lb, ub)
            new_fitness = np.apply_along_axis(func, 1, new_population)
            eval_count += len(new_fitness)

            # Selection: Replace if offspring are better
            better_indices = new_fitness < self.fitness
            self.population[better_indices] = new_population[better_indices]
            self.fitness[better_indices] = new_fitness[better_indices]

            # Update quantum probabilities
            best_idx = np.argmin(self.fitness)
            best_solution = self.population[best_idx]
            self.update_quantum_probabilities(best_solution)

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]