import numpy as np

class HybridDE:
    def __init__(self, budget=10000, dim=10, population_size=50, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.F = F  # Differential weight
        self.CR = CR  # Crossover probability
        self.f_opt = np.Inf
        self.x_opt = None
        self.cma_covariance = np.eye(dim)
        
    def evolve_population(self, pop, func):
        new_pop = np.copy(pop)
        for i in range(self.population_size):
            idxs = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = a + self.F * (b - c)
            mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)
            trial = np.copy(pop[i])
            for j in range(self.dim):
                if np.random.rand() < self.CR:
                    trial[j] = mutant[j]
            f_trial = func(trial)
            if f_trial < func(pop[i]):
                new_pop[i] = trial
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial
        return new_pop

    def parameter_adaptation(self, generation):
        self.F = 0.5 + 0.3 * np.cos(np.pi * generation / self.budget)
        self.CR = 0.9 * (1 - np.exp(-generation / 200))

    def covariance_update(self, pop):
        mean = np.mean(pop, axis=0)
        self.cma_covariance = np.cov(pop.T) + 0.01 * np.eye(self.dim)
        for i in range(self.population_size):
            pop[i] += np.random.multivariate_normal(np.zeros(self.dim), self.cma_covariance)
            pop[i] = np.clip(pop[i], func.bounds.lb, func.bounds.ub)
        return pop

    def __call__(self, func):
        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.population_size, self.dim))
        evals = 0
        while evals < self.budget:
            pop = self.evolve_population(pop, func)
            pop = self.covariance_update(pop)
            evals += self.population_size
            self.parameter_adaptation(evals // self.population_size)
        return self.f_opt, self.x_opt