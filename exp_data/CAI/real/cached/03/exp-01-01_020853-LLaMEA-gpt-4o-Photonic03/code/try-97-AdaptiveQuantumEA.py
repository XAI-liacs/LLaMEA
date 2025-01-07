import numpy as np

class AdaptiveQuantumEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.q_population = np.random.rand(self.population_size, dim)
        self.best_solution = None
        self.best_fitness = float('inf')
        self.mutation_rate = 0.1  # New mutation rate parameter

    def measure(self, q_individual):
        return np.round(q_individual)

    def evaluate_population(self, func):
        classical_population = np.array([self.measure(q_ind) for q_ind in self.q_population])
        fitness = np.array([func(ind) for ind in classical_population])
        return fitness, classical_population

    def update_quantum_population(self, fitness, classical_population):
        sorted_indices = np.argsort(fitness)
        best_indices = sorted_indices[:self.population_size // 2]
        for i in range(self.population_size):
            if i not in best_indices:
                best_individual = classical_population[np.random.choice(best_indices)]
                learning_rate = 0.5 * (1 - fitness[i] / np.max(fitness))  # Adaptive learning rate
                self.q_population[i] += learning_rate * (best_individual - self.q_population[i])
                self.q_population[i] = np.clip(self.q_population[i], 0, 1)
        
        random_indices = np.random.choice(self.population_size, self.population_size // 10, replace=False)
        self.q_population[random_indices] = np.random.rand(len(random_indices), self.dim)
        
        # Apply mutation to the population
        mutation_mask = np.random.rand(*self.q_population.shape) < self.mutation_rate
        self.q_population += mutation_mask * np.random.normal(0, 0.1, self.q_population.shape)

    def __call__(self, func):
        func_bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        evaluations = 0

        while evaluations < self.budget:
            fitness, classical_population = self.evaluate_population(func)
            evaluations += len(fitness)

            min_fitness_index = np.argmin(fitness)
            if fitness[min_fitness_index] < self.best_fitness:
                self.best_fitness = fitness[min_fitness_index]
                self.best_solution = classical_population[min_fitness_index]

            if evaluations >= self.budget:
                break

            self.update_quantum_population(fitness, classical_population)

        real_solution = self.best_solution * (func_bounds[:, 1] - func_bounds[:, 0]) + func_bounds[:, 0]
        return real_solution