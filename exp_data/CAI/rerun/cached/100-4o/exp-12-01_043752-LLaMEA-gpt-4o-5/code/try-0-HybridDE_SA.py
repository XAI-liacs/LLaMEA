import numpy as np

class HybridDE_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.fitness = np.array([float('inf')] * self.pop_size)
        self.best_solution = None
        self.best_fitness = float('inf')

    def __call__(self, func):
        eval_count = 0

        # Evaluate initial population
        for i in range(self.pop_size):
            self.fitness[i] = func(self.population[i])
            eval_count += 1
            if self.fitness[i] < self.best_fitness:
                self.best_fitness = self.fitness[i]
                self.best_solution = self.population[i].copy()

        while eval_count < self.budget:
            for i in range(self.pop_size):
                # Mutation
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                mutant_vector = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)

                # Crossover
                crossover_vector = np.where(
                    np.random.rand(self.dim) < self.CR, 
                    mutant_vector, 
                    self.population[i]
                )

                # Simulated Annealing acceptance
                new_fitness = func(crossover_vector)
                eval_count += 1
                if eval_count >= self.budget:
                    break

                if new_fitness < self.fitness[i] or np.random.rand() < np.exp(-(new_fitness - self.fitness[i]) / (1 + eval_count / self.budget)):
                    self.population[i] = crossover_vector
                    self.fitness[i] = new_fitness

                    if new_fitness < self.best_fitness:
                        self.best_fitness = new_fitness
                        self.best_solution = crossover_vector.copy()

        return self.best_solution