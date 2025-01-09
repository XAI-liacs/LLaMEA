import numpy as np

class QIDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.F = 0.5
        self.CR = 0.9
        self.qubit_rotation_angle = np.pi / 4

    def rotate(self, x):
        return np.cos(self.qubit_rotation_angle) * x - np.sin(self.qubit_rotation_angle) * x

    def differential_evolution(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        fitness = [func(individual) for individual in population]
        
        while self.budget > 0:
            for i in range(self.population_size):
                idxs = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = population[idxs]
                mutant = population[a] + self.F * (population[b] - population[c])
                mutant = np.clip(mutant, -5.0, 5.0)
                
                crossed = np.array([mutant[j] if np.random.rand() < self.CR or j == np.random.randint(0, self.dim) else population[i, j] for j in range(self.dim)])
                crossed_fitness = func(crossed)
                if crossed_fitness < fitness[i]:
                    population[i] = crossed
                    fitness[i] = crossed_fitness

                self.budget -= 1
                if self.budget == 0:
                    break
        
        best_idx = np.argmin(fitness)
        return population[best_idx]

    def __call__(self, func):
        return self.differential_evolution(func)