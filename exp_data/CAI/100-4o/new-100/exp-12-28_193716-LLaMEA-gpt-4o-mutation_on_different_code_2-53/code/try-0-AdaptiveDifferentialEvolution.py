import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 10 * dim
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.bounds = None

    def __call__(self, func):
        self.bounds = func.bounds
        lb, ub = self.bounds.lb, self.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        pop_fitness = np.array([func(ind) for ind in pop])
        evaluations = self.population_size
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                idxs = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = pop[idxs]
                mutant = np.clip(a + self.F * (b - c), lb, ub)
                
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, pop[i])
                trial_fitness = func(trial)
                evaluations += 1
                
                if trial_fitness < pop_fitness[i]:
                    pop[i] = trial
                    pop_fitness[i] = trial_fitness
                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = trial

                if evaluations >= self.budget:
                    break
        
        return self.f_opt, self.x_opt