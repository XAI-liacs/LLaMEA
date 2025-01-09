import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, population_size=50, F=0.8, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.F = F  # Mutation factor
        self.CR = CR  # Crossover probability
        self.f_opt = np.Inf
        self.x_opt = None
        self.func_evals = 0

    def mutate(self, population, target_idx):
        indices = list(range(self.population_size))
        indices.remove(target_idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant_vector = population[a] + self.F * (population[b] - population[c])
        return np.clip(mutant_vector, -5, 5)

    def crossover(self, target, mutant):
        crossover_vector = np.where(np.random.rand(self.dim) < self.CR, mutant, target)
        return crossover_vector

    def select(self, target, trial, func):
        trial_fitness = func(trial)
        self.func_evals += 1
        if trial_fitness < self.f_opt:
            self.f_opt = trial_fitness
            self.x_opt = trial
        target_fitness = func(target)
        self.func_evals += 1
        return trial if trial_fitness < target_fitness else target

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        while self.func_evals < self.budget:
            for i in range(self.population_size):
                if self.func_evals >= self.budget:
                    break
                target = population[i]
                mutant = self.mutate(population, i)
                trial = self.crossover(target, mutant)
                population[i] = self.select(target, trial, func)
        return self.f_opt, self.x_opt