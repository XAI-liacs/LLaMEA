import numpy as np

class HybridDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = 20
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        self.evaluate_population()

    def levy_flight(self, beta=1.5):
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, size=(self.dim,))
        v = np.random.normal(0, 1, size=(self.dim,))
        step = u / np.abs(v) ** (1 / beta)
        return step

    def evaluate_population(self):
        self.fitness = np.array([np.inf] * self.population_size)

    def __call__(self, func):
        evals = 0
        for i in range(self.population_size):
            self.fitness[i] = func(self.population[i])
            evals += 1
            if evals >= self.budget:
                return self.best_solution()

        while evals < self.budget:
            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
                mutant = a + self.mutation_factor * (b - c)

                for d in range(self.dim):
                    if np.random.rand() > self.crossover_rate:
                        mutant[d] = self.population[i][d]

                mutant = np.clip(mutant, self.bounds[0], self.bounds[1])

                mutant += self.levy_flight() * (self.bounds[1] - self.bounds[0]) * 0.01
                mutant = np.clip(mutant, self.bounds[0], self.bounds[1])

                mutant_fitness = func(mutant)
                evals += 1

                if mutant_fitness < self.fitness[i]:
                    self.population[i] = mutant
                    self.fitness[i] = mutant_fitness

                if evals >= self.budget:
                    return self.best_solution()

        return self.best_solution()

    def best_solution(self):
        idx = np.argmin(self.fitness)
        return self.population[idx], self.fitness[idx]