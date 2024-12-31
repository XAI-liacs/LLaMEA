import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, population_size=30, F=0.5, CR=0.9, F_min=0.2, F_max=0.8, F_scale=0.1):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.F = F
        self.F_min = F_min
        self.F_max = F_max
        self.F_scale = F_scale
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        for i in range(self.budget):
            diversity = np.mean(np.std(population, axis=0))
            self.F = max(self.F_min, min(self.F_max, self.F + self.F_scale * (diversity - np.median(diversity))))
            
            for j in range(self.population_size):
                idxs = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = population[idxs]
                mutant = a + self.F * (b - c)
                cross_points = np.random.rand(self.dim) < self.CR
                trial = np.where(cross_points, mutant, population[j])
                
                f_trial = func(trial)
                if f_trial < fitness[j]:
                    population[j] = trial
                    fitness[j] = f_trial
                
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial
            
        return self.f_opt, self.x_opt