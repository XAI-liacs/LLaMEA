import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, population_size=50, f_min=0.1, f_max=0.9, cr_min=0.1, cr_max=0.9):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = population_size
        self.f_min = f_min
        self.f_max = f_max
        self.cr_min = cr_min
        self.cr_max = cr_max

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)]
        
        # Adaptive parameters
        F = np.random.uniform(self.f_min, self.f_max, self.population_size)
        CR = np.random.uniform(self.cr_min, self.cr_max, self.population_size)
        
        evaluations = self.population_size
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                # Mutation
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + F[i] * (b - c), func.bounds.lb, func.bounds.ub)
                
                # Crossover
                cross_points = np.random.rand(self.dim) < CR[i]
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, population[i])
                
                # Selection
                f_trial = func(trial)
                evaluations += 1
                
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    F[i] = np.clip(F[i] + 0.1, self.f_min, self.f_max)  # Increase mutation scale
                    CR[i] = np.clip(CR[i] + 0.1, self.cr_min, self.cr_max)  # Increase crossover rate
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                else:
                    F[i] *= 0.95  # Gradual reduction in mutation scale
                    CR[i] *= 0.95  # Gradual reduction in crossover rate
                
                if evaluations >= self.budget:
                    break
            
        return self.f_opt, self.x_opt