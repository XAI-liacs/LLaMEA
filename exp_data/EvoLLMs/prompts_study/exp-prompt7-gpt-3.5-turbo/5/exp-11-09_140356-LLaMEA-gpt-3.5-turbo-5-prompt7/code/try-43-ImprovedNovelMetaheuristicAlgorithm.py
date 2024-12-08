import numpy as np

class ImprovedNovelMetaheuristicAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.best_solution = self.population[0]

    def __call__(self, func):
        for _ in range(self.budget):
            # Select parents
            idx = np.argsort([func(x) for x in self.population])
            parent1, parent2 = self.population[idx[0]], self.population[idx[1]]

            # Novel mutation strategy
            beta = np.random.uniform(0.5, 1.0, self.dim)
            offspring = parent1 + beta * (parent2 - self.population)

            # Replace worst solution
            idx_worst = np.argmax([func(x) for x in self.population])
            self.population[idx_worst] = offspring

            # Elitism mechanism
            if func(offspring) < func(self.best_solution):
                self.best_solution = offspring

        return self.best_solution