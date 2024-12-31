import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=20):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        # Initialize population
        bounds = func.bounds
        lb, ub = bounds.lb, bounds.ub
        pop = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        
        # Initialize DE parameters
        F = 0.5   # Mutation factor
        CR = 0.9  # Crossover rate
        
        evals = self.pop_size
        while evals < self.budget:
            for i in range(self.pop_size):
                # Mutation (rand/1 strategy)
                indices = np.random.choice(self.pop_size, 3, replace=False)
                x1, x2, x3 = pop[indices]
                mutant = np.clip(x1 + F * (x2 - x3), lb, ub)
                
                # Crossover
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                
                # Selection
                f_trial = func(trial)
                evals += 1
                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    pop[i] = trial
                    
                # Update best solution
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial

                # Dynamic adaptation of F and CR
                F = np.clip(0.5 + 0.3 * np.random.standard_normal(), 0.1, 0.9)
                CR = np.clip(0.9 + 0.1 * np.random.standard_normal(), 0.5, 1.0)
                
                # Early stopping if budget is exhausted
                if evals >= self.budget:
                    break
        
        return self.f_opt, self.x_opt