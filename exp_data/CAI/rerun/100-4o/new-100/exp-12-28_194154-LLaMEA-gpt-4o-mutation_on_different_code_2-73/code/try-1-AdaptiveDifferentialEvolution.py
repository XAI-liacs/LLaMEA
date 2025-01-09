import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, population_size=50, F=0.8, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None
        self.history = []

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.population_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.history.append(fitness.min())
        
        for _ in range(self.budget - self.population_size):
            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), func.bounds.lb, func.bounds.ub)
                
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                # Enhanced mutation using the best solution found so far
                trial = np.where(cross_points, mutant, self.x_opt if self.x_opt is not None else population[i])
                f = func(trial)
                
                if f < fitness[i]:
                    population[i] = trial
                    fitness[i] = f
                    if f < self.f_opt:
                        self.f_opt = f
                        self.x_opt = trial
            
            current_best = fitness.min()
            if len(self.history) > 1:
                improvement = (self.history[-1] - current_best) / self.history[-1]
                self.F = np.clip(self.F * (1 + improvement), 0.5, 1.0)
                self.CR = np.clip(self.CR * (1 - improvement), 0.6, 1.0)
            self.history.append(current_best)
        
        return self.f_opt, self.x_opt