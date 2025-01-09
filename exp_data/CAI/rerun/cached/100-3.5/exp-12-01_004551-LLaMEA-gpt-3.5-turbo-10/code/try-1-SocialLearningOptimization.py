import numpy as np

class SocialLearningOptimization:
    def __init__(self, budget, dim, pop_size=30, alpha=0.1, beta=0.5):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.alpha = alpha
        self.beta = beta

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        for _ in range(self.budget):
            idx = np.argsort(fitness)
            best_individual = population[idx[0]]
            for i in range(self.pop_size):
                r1, r2 = np.random.choice(np.delete(np.arange(self.pop_size), i), 2, replace=False)
                new_individual = population[i] + self.alpha * (best_individual - population[i]) + self.beta * (population[r1] - population[r2])
                new_fitness = func(new_individual)
                if new_fitness < fitness[i]:
                    population[i] = new_individual
                    fitness[i] = new_fitness
        return population[idx[0]]