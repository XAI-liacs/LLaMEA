import numpy as np

class ImprovedQIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.learning_rate = 0.1

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(individual) for individual in self.population]
            sorted_indices = np.argsort(fitness)
            elite_index = sorted_indices[0]
            elite = self.population[elite_index]
            for i in range(self.budget):
                if i != elite_index:
                    mutation = self.population[i] + np.random.uniform(-0.1, 0.1, self.dim)
                    offspring = 0.5 * self.population[i] + 0.5 * elite
                    quantum_mutation = offspring + np.random.choice([-1, 1]) * np.random.rand() * self.learning_rate * (elite - self.population[i])
                    self.population[i] = quantum_mutation if func(quantum_mutation) < fitness[i] else mutation
            self.learning_rate *= 0.995  # Decay the learning rate for dynamic adaptation
        best_index = np.argmin([func(individual) for individual in self.population])
        return self.population[best_index]