import numpy as np

class Hybrid_Quantum_Genetic_Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 40  # Increased for better exploration
        self.q_factor = 0.7  # Quantum exploration factor
        self.crossover_rate = 0.5
        self.mutation_rate = 0.1  # Higher mutation rate for diversity
        self.adaptation_factor = 0.95  # Adaptive control for exploration and exploitation balance

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        np.random.seed(42)

        # Initialize population
        position = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(p) for p in position])
        evaluations = self.population_size
        
        best_indices = np.argsort(fitness)
        global_best_position = position[best_indices[0]]
        global_best_value = fitness[best_indices[0]]

        while evaluations < self.budget:
            # Quantum-inspired exploration
            quantum_shift = self.q_factor * np.random.normal(size=(self.population_size, self.dim))
            position += quantum_shift * np.random.uniform(-1, 1, (self.population_size, self.dim))
            position = np.clip(position, lb, ub)

            # Evaluate new positions
            new_fitness = np.array([func(p) for p in position])
            evaluations += self.population_size

            # Select the best positions for the next generation
            combined_positions = np.vstack((position, position[best_indices]))
            combined_fitness = np.hstack((new_fitness, fitness[best_indices]))

            best_indices = np.argsort(combined_fitness)[:self.population_size]
            position = combined_positions[best_indices]
            fitness = combined_fitness[best_indices]

            # Update global best
            if fitness[0] < global_best_value:
                global_best_position = position[0]
                global_best_value = fitness[0]

            # Adaptive crossover and mutation
            new_population = []
            for i in range(0, self.population_size, 2):
                if np.random.rand() < self.crossover_rate:
                    parent1, parent2 = position[i], position[(i + 1) % self.population_size]
                    offspring1, offspring2 = self.crossover(parent1, parent2)
                    new_population.extend([offspring1, offspring2])
                else:
                    new_population.extend([position[i], position[(i + 1) % self.population_size]])

            # Apply mutation
            for i in range(self.population_size):
                if np.random.rand() < self.mutation_rate:
                    new_population[i] += self.mutation_rate * np.random.normal(size=self.dim)
                    new_population[i] = np.clip(new_population[i], lb, ub)

            position = np.array(new_population)
            fitness = np.array([func(p) for p in position])
            evaluations += self.population_size
            
            # Adapt exploration parameters
            self.q_factor *= self.adaptation_factor

        return global_best_position, global_best_value

    def crossover(self, parent1, parent2):
        alpha = np.random.rand(self.dim)
        offspring1 = alpha * parent1 + (1 - alpha) * parent2
        offspring2 = alpha * parent2 + (1 - alpha) * parent1
        return offspring1, offspring2