import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, NP=30):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.NP = NP
        self.f_opt = np.Inf
        self.x_opt = None

    def evolve_population(self, population, func):
        for i in range(self.NP):
            target = population[i]
            indices = [idx for idx in range(self.NP) if idx != i]
            a, b, c = population[np.random.choice(indices, 3, replace=False)]
            mutant = np.clip(a + self.F * (b - c), func.bounds.lb, func.bounds.ub)
            
            crossover_points = np.random.rand(self.dim) < self.CR
            trial = np.where(crossover_points, mutant, target)
            
            f_target = func(target)
            f_trial = func(trial)
            if f_trial < f_target:
                population[i] = trial
        
        return population

    def __call__(self, func):
        population = [np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim) for _ in range(self.NP)]
        
        for _ in range(self.budget):
            population = self.evolve_population(population, func)
            best_idx = np.argmin([func(individual) for individual in population])
            best_individual = population[best_idx]
            if func(best_individual) < self.f_opt:
                self.f_opt = func(best_individual)
                self.x_opt = best_individual
        
        return self.f_opt, self.x_opt