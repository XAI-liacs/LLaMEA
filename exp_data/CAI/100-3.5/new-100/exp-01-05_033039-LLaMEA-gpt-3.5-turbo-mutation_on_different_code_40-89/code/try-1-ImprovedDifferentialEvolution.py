import numpy as np

class ImprovedDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, p=0.2):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.p = p
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        fitness = np.array([func(x) for x in population])

        for i in range(self.budget):
            for j in range(len(population)):
                idxs = [idx for idx in range(len(population)) if idx != j]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), func.bounds.lb, func.bounds.ub)
                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, mutant, population[j])
                f_trial = func(trial)
                
                if f_trial < fitness[j]:
                    fitness[j] = f_trial
                    population[j] = trial
                
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial
                    
            # Introduce population diversity preservation
            mean_population = np.mean(population, axis=0)
            for j in range(len(population)):
                if np.random.rand() < self.p:
                    population[j] = population[j] + np.random.normal(0, 1, self.dim) * mean_population

        return self.f_opt, self.x_opt