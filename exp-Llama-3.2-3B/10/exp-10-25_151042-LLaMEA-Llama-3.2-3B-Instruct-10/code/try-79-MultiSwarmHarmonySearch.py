import numpy as np
import random
from scipy.optimize import minimize

class MultiSwarmHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm = self.initialize_swarm()
        self.best_solution = None
        self.best_fitness = float('inf')
        self.mutation_probability = 0.1

    def initialize_swarm(self):
        swarm = []
        for _ in range(self.budget):
            solution = np.random.uniform(-5.0, 5.0, self.dim)
            swarm.append(solution)
        return swarm

    def fitness(self, solution):
        return func(solution)

    def __call__(self, func):
        for _ in range(self.budget):
            if self.best_fitness > func(self.swarm[0]):
                self.best_solution = self.swarm[0]
                self.best_fitness = func(self.swarm[0])

            for i, solution in enumerate(self.swarm):
                fitness = func(solution)
                if fitness < self.best_fitness:
                    self.swarm[i] = solution
                    self.best_fitness = fitness

            # Adaptation phase
            for i in range(len(self.swarm)):
                r1, r2 = random.random(), random.random()
                if r1 < 0.2:
                    self.swarm[i] = self.swarm[i] + 0.1 * (self.swarm[i] - self.best_solution)
                elif r2 < 0.4:
                    self.swarm[i] = self.swarm[i] - 0.1 * (self.swarm[i] - self.best_solution)
                elif r1 < 0.6:
                    self.swarm[i] = self.swarm[i] + 0.1 * np.random.uniform(-1, 1, self.dim)
                elif r2 < 0.8:
                    self.swarm[i] = self.swarm[i] - 0.1 * np.random.uniform(-1, 1, self.dim)

            # Mutation phase
            if random.random() < self.mutation_probability:
                r1, r2 = random.random(), random.random()
                if r1 < 0.5:
                    mutation = np.random.uniform(-1, 1, self.dim)
                    self.swarm[i] = self.swarm[i] + mutation
                else:
                    mutation = np.random.uniform(-1, 1, self.dim)
                    self.swarm[i] = self.swarm[i] - mutation

        return self.best_solution, self.best_fitness

def func(solution):
    # Example noiseless function
    return np.sum(solution**2)

# Example usage
ms = MultiSwarmHarmonySearch(100, 10)
best_solution, best_fitness = ms(func)
print(f"Best solution: {best_solution}")
print(f"Best fitness: {best_fitness}")