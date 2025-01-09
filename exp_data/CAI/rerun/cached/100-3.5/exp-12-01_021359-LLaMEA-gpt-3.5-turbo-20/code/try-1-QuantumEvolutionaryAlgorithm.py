import numpy as np

class QuantumEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
    
    def __call__(self, func):
        best_solution = None
        best_fitness = np.inf
        for _ in range(self.budget):
            fitness_values = np.array([func(individual) for individual in self.population])
            best_idx = np.argmin(fitness_values)
            if fitness_values[best_idx] < best_fitness:
                best_solution = self.population[best_idx]
                best_fitness = fitness_values[best_idx]
            pivot = np.random.randint(0, self.dim)
            for idx, individual in enumerate(self.population):
                if idx != best_idx:
                    alpha = np.random.random()
                    self.population[idx] = alpha * individual + (1 - alpha) * self.population[best_idx]
                    if np.random.random() < 0.5:
                        self.population[idx][pivot] = np.pi - individual[pivot]
        return best_solution