import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, population_size=50):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.population_size = population_size
        self.f_opt = np.Inf
        self.x_opt = None

    def mutate(self, population, target_index):
        candidates = [idx for idx in range(self.population_size) if idx != target_index]
        a, b, c = np.random.choice(candidates, 3, replace=False)
        mutant_vector = population[a] + self.F * (population[b] - population[c])
        return mutant_vector

    def crossover(self, target_vector, mutant_vector):
        crossover_points = np.random.rand(self.dim) < self.CR
        trial_vector = np.where(crossover_points, mutant_vector, target_vector)
        return trial_vector

    def selection(self, func, target_vector, trial_vector):
        target_fitness = func(target_vector)
        trial_fitness = func(trial_vector)
        if trial_fitness < target_fitness:
            return trial_vector, trial_fitness
        else:
            return target_vector, target_fitness

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        for _ in range(self.budget):
            for i in range(self.population_size):
                target_vector = population[i]
                mutant_vector = self.mutate(population, i)
                trial_vector = self.crossover(target_vector, mutant_vector)
                population[i], fitness = self.selection(func, target_vector, trial_vector)
                if fitness < self.f_opt:
                    self.f_opt = fitness
                    self.x_opt = population[i]

        return self.f_opt, self.x_opt