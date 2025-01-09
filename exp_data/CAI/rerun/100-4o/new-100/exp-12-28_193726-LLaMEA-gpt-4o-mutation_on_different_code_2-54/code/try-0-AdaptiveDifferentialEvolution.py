import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=None):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size if pop_size is not None else 10 * dim
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        # Keep track of best solution
        best_idx = np.argmin(fitness)
        self.x_opt = population[best_idx]
        self.f_opt = fitness[best_idx]
        
        F = 0.5  # mutation factor
        CR = 0.9  # crossover probability
        evals = len(population)
        
        while evals < self.budget:
            for i in range(self.pop_size):
                # Mutation
                idxs = np.random.choice(self.pop_size, 3, replace=False)
                x1, x2, x3 = population[idxs]
                mutant = np.clip(x1 + F * (x2 - x3), func.bounds.lb, func.bounds.ub)
                
                # Crossover
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                # Selection
                f_trial = func(trial)
                evals += 1
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                
                # Elitism
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial
                
                if evals >= self.budget:
                    break
            
            # Adaptive control parameter update
            CR = 0.9 - 0.5 * (evals / self.budget)  # Decrease CR over time
            F = 0.5 + 0.2 * np.random.rand()  # Add randomness to F

        return self.f_opt, self.x_opt