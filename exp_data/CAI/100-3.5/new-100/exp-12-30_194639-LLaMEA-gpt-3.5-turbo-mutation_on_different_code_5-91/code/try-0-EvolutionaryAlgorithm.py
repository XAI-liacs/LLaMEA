import numpy as np

class EvolutionaryAlgorithm:
    def __init__(self, budget=10000, dim=10, pop_size=50, cr=0.9, f=0.8):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.cr = cr
        self.f = f
        self.population = np.random.uniform(-5.0, 5.0, (pop_size, dim))
        self.fitness = np.full(pop_size, np.Inf)
        self.f_opt = np.Inf
        self.x_opt = None

    def mutate(self, target, pop, idx):
        candidates = np.random.choice(self.pop_size, 3, replace=False)
        mutant = pop[candidates[0]] + self.f * (pop[candidates[1]] - pop[candidates[2]])
        cross_points = np.random.rand(self.dim) < self.cr
        cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def __call__(self, func):
        for _ in range(self.budget // self.pop_size):
            for i in range(self.pop_size):
                fitness = func(self.population[i])
                if fitness < self.fitness[i]:
                    self.fitness[i] = fitness
                    if fitness < self.f_opt:
                        self.f_opt = fitness
                        self.x_opt = self.population[i].copy()

                trial = self.mutate(self.population[i], self.population, i)
                trial_fitness = func(trial)
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = trial.copy()

        return self.f_opt, self.x_opt