import numpy as np

class QuantumInspiredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50  # Quantum bit individuals
        self.quantum_prob = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.best_solution = None
        self.best_fitness = float('inf')
        self.evaluations = 0
        self.history = []

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub

        def evaluate_and_update(population):
            fitness = np.array([func(x) for x in population])
            min_fitness_idx = np.argmin(fitness)
            nonlocal self
            if fitness[min_fitness_idx] < self.best_fitness:
                self.best_fitness = fitness[min_fitness_idx]
                self.best_solution = population[min_fitness_idx].copy()
            return fitness

        def generate_population():
            population = np.random.uniform(lb, ub, (self.population_size, self.dim))
            for i in range(self.population_size):
                for j in range(self.dim):
                    if np.random.rand() < abs(self.quantum_prob[i][j]):
                        population[i][j] = ub[j]
                    else:
                        population[i][j] = lb[j]
            return np.clip(population, lb, ub)

        while self.evaluations < self.budget:
            population = generate_population()
            fitness = evaluate_and_update(population)
            self.evaluations += self.population_size

            # Adaptive quantum gate application
            for i in range(self.population_size):
                for j in range(self.dim):
                    if np.random.rand() < 0.5:  # Adaptive decision influenced by solution quality
                        theta = np.arccos(self.quantum_prob[i][j])
                        if fitness[i] < np.mean(fitness):
                            self.quantum_prob[i][j] = np.cos(theta + np.random.uniform(0, np.pi/4))
                        else:
                            self.quantum_prob[i][j] = np.cos(theta - np.random.uniform(0, np.pi/4))

            # Record the history of best solutions
            self.history.append(self.best_solution)

        return self.best_solution