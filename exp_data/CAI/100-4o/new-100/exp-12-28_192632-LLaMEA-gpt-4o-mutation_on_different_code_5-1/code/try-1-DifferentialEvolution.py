import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, population_size=20, F=0.8, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.F = F  # Differential weight
        self.CR = CR  # Crossover probability
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        population = np.random.uniform(bounds[0], bounds[1], (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])

        for evals in range(self.population_size, self.budget):
            for i in range(self.population_size):
                # Adjust CR based on evaluations
                dynamic_CR = self.CR * (1 - evals / self.budget)
                
                # Mutation
                idxs = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = population[idxs]
                mutant = np.clip(a + self.F * (b - c), bounds[0], bounds[1])

                # Crossover
                crossover = np.random.rand(self.dim) < dynamic_CR
                trial = np.where(crossover, mutant, population[i])

                # Selection
                f_trial = func(trial)
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial

                # Track the best solution
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial

                evals += 1
                if evals >= self.budget:
                    break

        return self.f_opt, self.x_opt