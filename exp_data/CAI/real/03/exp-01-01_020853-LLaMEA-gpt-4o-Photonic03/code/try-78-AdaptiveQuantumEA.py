import numpy as np

class AdaptiveQuantumEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50  # Initial population size
        self.q_population = np.random.rand(self.population_size, dim)  # Quantum-inspired encoding
        self.best_solution = None
        self.best_fitness = float('inf')

    def measure(self, q_individual):
        # Measure quantum individual to classical solution
        return np.round(q_individual)

    def evaluate_population(self, func):
        classical_population = np.array([self.measure(q_ind) for q_ind in self.q_population])
        fitness = np.array([func(ind) for ind in classical_population])
        return fitness, classical_population

    def update_quantum_population(self, fitness, classical_population):
        sorted_indices = np.argsort(fitness)
        best_indices = sorted_indices[:self.population_size // 2]  # Top 50% individuals
        for i in range(self.population_size):
            # Update rule inspired by best individuals
            if i not in best_indices:
                best_individual = classical_population[np.random.choice(best_indices)]
                self.q_population[i] += 0.4 * (best_individual - self.q_population[i])  # Adjusted influence
                self.q_population[i] = np.clip(self.q_population[i], 0, 1)

        # Change 1: Randomize a portion of the population to enhance exploration
        random_indices = np.random.choice(self.population_size, self.population_size // 10, replace=False)
        self.q_population[random_indices] = np.random.rand(len(random_indices), self.dim)
        
        # Additional randomization for exploration enhancement
        another_random_indices = np.random.choice(self.population_size, self.population_size // 15, replace=False)
        self.q_population[another_random_indices] = np.random.rand(len(another_random_indices), self.dim)

    def __call__(self, func):
        func_bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        evaluations = 0

        while evaluations < self.budget:
            fitness, classical_population = self.evaluate_population(func)
            evaluations += len(fitness)

            # Update the best solution found
            min_fitness_index = np.argmin(fitness)
            if fitness[min_fitness_index] < self.best_fitness:
                self.best_fitness = fitness[min_fitness_index]
                self.best_solution = classical_population[min_fitness_index]

            if evaluations >= self.budget:
                break

            self.update_quantum_population(fitness, classical_population)

        # Convert solution to real-world scale within bounds
        real_solution = self.best_solution * (func_bounds[:, 1] - func_bounds[:, 0]) + func_bounds[:, 0]
        return real_solution