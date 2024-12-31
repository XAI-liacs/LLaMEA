import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, population_size=50, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.F = F  # Differential weight
        self.CR = CR  # Crossover probability
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        pop = np.random.uniform(bounds[0], bounds[1], (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        
        for evals in range(self.population_size, self.budget):
            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                
                # Mutation and crossover
                F_noise = np.random.normal(0, 0.1)  # Add noise to differential weight
                mutant = np.clip(pop[a] + (self.F + F_noise) * (pop[b] - pop[c]), bounds[0], bounds[1])
                cross_points = np.random.rand(self.dim) < self.CR
                trial = np.where(cross_points, mutant, pop[i])
                
                f_trial = func(trial)
                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    pop[i] = trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                    
            # Adaptive control parameters
            if evals % 100 == 0:  # Adaptive step every 100 evaluations
                success_rate = sum([1 for fit in fitness if fit < np.mean(fitness)]) / self.population_size
                self.F = np.clip(self.F + 0.1 * (0.5 - success_rate), 0.1, 0.9)
                self.CR = np.clip(self.CR + 0.1 * (0.2 - success_rate), 0.1, 1.0)
            
        return self.f_opt, self.x_opt