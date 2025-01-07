import numpy as np

class AdaptiveQuantumEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.population_size = self.initial_population_size
        self.q_population = np.random.rand(self.population_size, dim)
        self.best_solution = None
        self.best_fitness = float('inf')

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
                influence = 0.4 + (0.2 * np.random.rand())  # Dynamic influence
                self.q_population[i] += influence * (best_individual - self.q_population[i])
                self.q_population[i] = np.clip(self.q_population[i], 0, 1)
        
        random_indices = np.random.choice(self.population_size, self.population_size // 10, replace=False)
        self.q_population[random_indices] = np.random.rand(len(random_indices), self.dim)

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
            self.population_size = self.initial_population_size + evaluations // (self.budget // 10)  # Adaptive size

        real_solution = self.best_solution * (func_bounds[:, 1] - func_bounds[:, 0]) + func_bounds[:, 0]
        return real_solution