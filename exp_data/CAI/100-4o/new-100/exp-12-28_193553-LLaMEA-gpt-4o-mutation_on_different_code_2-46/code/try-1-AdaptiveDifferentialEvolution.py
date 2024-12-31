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
        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        
        for eval_count in range(self.pop_size, self.budget):
            # Adapt parameters F and CR based on diversity and current progress
            F = 0.5 + (np.std(fitness) / np.mean(fitness)) * 0.5
            CR = 0.9 - (np.std(fitness) / np.mean(fitness)) * 0.4

            # Differential mutation and crossover
            for i in range(self.pop_size):
                idxs = np.random.choice(self.pop_size, 3, replace=False)
                x1, x2, x3 = pop[idxs]
                mutant = np.clip(x1 + F * (x2 - x3), func.bounds.lb, func.bounds.ub)
                
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                
                f_trial = func(trial)
                eval_count += 1
                
                if f_trial < fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial

                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial

                if eval_count >= self.budget:
                    break
            
            # Introduce restart if population diversity is too low
            if np.std(fitness) < 1e-5 and self.pop_size > 4:
                self.pop_size = max(4, self.pop_size // 2)
                pop = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.pop_size, self.dim)) # Restart strategy
                fitness = np.array([func(ind) for ind in pop])

        return self.f_opt, self.x_opt