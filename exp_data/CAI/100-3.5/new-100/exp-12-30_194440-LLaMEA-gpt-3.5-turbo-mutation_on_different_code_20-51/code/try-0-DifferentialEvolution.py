import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.cr = 0.5
        self.f = 0.8
        self.population_size = 10

    def evolve_population(self, func, population):
        new_population = []
        for i in range(self.population_size):
            target_vector = population[i]
            indices = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = population[np.random.choice(indices, 3, replace=False)]
            mutant_vector = target_vector + self.f * (a - b)
            crossover_points = np.random.rand(self.dim) < self.cr
            trial_vector = np.where(crossover_points, mutant_vector, target_vector)
            if func(trial_vector) < func(target_vector):
                new_population.append(trial_vector)
            else:
                new_population.append(target_vector)
        return new_population

    def __call__(self, func):
        population = [np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim) for _ in range(self.population_size)]
        for _ in range(int(self.budget / self.population_size)):
            population = self.evolve_population(func, population)
            for vector in population:
                f_val = func(vector)
                if f_val < self.f_opt:
                    self.f_opt = f_val
                    self.x_opt = vector
        return self.f_opt, self.x_opt