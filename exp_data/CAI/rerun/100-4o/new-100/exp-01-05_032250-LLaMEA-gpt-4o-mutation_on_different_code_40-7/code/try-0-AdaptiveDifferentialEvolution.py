import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 20 * dim
        self.F = 0.8  # Initial mutation factor
        self.CR = 0.9  # Initial crossover probability
        self.bounds = [-5.0, 5.0]
        
    def mutate(self, a, b, c):
        return a + self.F * (b - c)

    def crossover(self, target, donor):
        return np.where(np.random.rand(self.dim) < self.CR, donor, target)

    def select(self, target, trial, func):
        f_target = func(target)
        f_trial = func(trial)
        if f_trial < f_target:
            return trial, f_trial
        else:
            return target, f_target

    def update_parameters(self, generation):
        # Dynamic adjustment
        self.F = 0.5 + 0.3 * np.cos(generation * np.pi / (2 * (self.budget // self.population_size)))
        self.CR = 0.3 + 0.7 * np.sin(generation * np.pi / (2 * (self.budget // self.population_size)))

    def __call__(self, func):
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)]

        generation = 0
        while generation < self.budget // self.population_size:
            self.update_parameters(generation)
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                donor = self.mutate(a, b, c)
                donor = np.clip(donor, self.bounds[0], self.bounds[1])
                trial = self.crossover(population[i], donor)
                population[i], fitness[i] = self.select(population[i], trial, func)
                if fitness[i] < self.f_opt:
                    self.f_opt = fitness[i]
                    self.x_opt = population[i]
            generation += 1

        return self.f_opt, self.x_opt