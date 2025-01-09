import numpy as np

class PSO_ADE:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.pop_size = 50
        self.bounds = (-5.0, 5.0)
        self.c1 = 2.0
        self.c2 = 2.0
        self.w = 0.5
        self.F = 0.8
        self.CR = 0.9

    def __call__(self, func):
        lb, ub = self.bounds
        pop = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        velocities = np.zeros((self.pop_size, self.dim))
        p_best = pop.copy()
        p_best_values = np.array([func(x) for x in p_best])
        g_best = p_best[np.argmin(p_best_values)]
        
        eval_count = self.pop_size
        
        while eval_count < self.budget:
            # PSO update
            r1, r2 = np.random.rand(self.pop_size, self.dim), np.random.rand(self.pop_size, self.dim)
            velocities = (self.w * velocities + 
                          self.c1 * r1 * (p_best - pop) + 
                          self.c2 * r2 * (g_best - pop))
            pop = np.clip(pop + velocities, lb, ub)
            
            # Evaluate population
            pop_values = np.array([func(x) for x in pop])
            eval_count += self.pop_size
            
            # Update personal and global best
            better_mask = pop_values < p_best_values
            p_best[better_mask] = pop[better_mask]
            p_best_values[better_mask] = pop_values[better_mask]
            if np.min(p_best_values) < func(g_best):
                g_best = p_best[np.argmin(p_best_values)]
            
            # ADE mutation
            for i in range(self.pop_size):
                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), lb, ub)
                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, mutant, pop[i])
                
                trial_value = func(trial)
                eval_count += 1
                
                if trial_value < pop_values[i]:
                    pop[i] = trial
                    pop_values[i] = trial_value
                    if trial_value < p_best_values[i]:
                        p_best[i] = trial
                        p_best_values[i] = trial_value
                        if trial_value < func(g_best):
                            g_best = trial

            if eval_count >= self.budget:
                break

        self.f_opt = func(g_best)
        self.x_opt = g_best
        return self.f_opt, self.x_opt