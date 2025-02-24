import numpy as np
from scipy.optimize import minimize

class PSOAdaptivePeriodicityOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = None
        self.ub = None
    
    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        swarm_size = 20 + 2 * self.dim
        w, c1, c2 = 0.5, 1.5, 1.5
        x = self.initialize_population(swarm_size)
        v = np.random.uniform(-1, 1, (swarm_size, self.dim))
        p_best = x.copy()
        p_best_fitness = np.array([func(ind) for ind in p_best])
        g_best_idx = np.argmin(p_best_fitness)
        g_best = p_best[g_best_idx]
        eval_count = swarm_size
        
        while eval_count < self.budget:
            for i in range(swarm_size):
                if eval_count >= self.budget:
                    break

                # Update velocity
                r1, r2 = np.random.rand(2)
                v[i] = (w * v[i] + 
                        c1 * r1 * (p_best[i] - x[i]) + 
                        c2 * r2 * (g_best - x[i]))

                # Update position
                x[i] = np.clip(x[i] + v[i], self.lb, self.ub)

                # Enforce periodicity
                x[i] = self.adaptive_enforce_periodicity(x[i])

                # Evaluate new position
                new_fitness = func(x[i])
                eval_count += 1

                # Update personal best
                if new_fitness < p_best_fitness[i]:
                    p_best[i] = x[i]
                    p_best_fitness[i] = new_fitness

                    # Update global best
                    if new_fitness < p_best_fitness[g_best_idx]:
                        g_best = x[i]
                        g_best_idx = i

            # Local refinement with periodicity enhancement
            if eval_count < self.budget:
                result = minimize(func, g_best, bounds=list(zip(self.lb, self.ub)), method='L-BFGS-B')
                eval_count += result.nfev
                if result.fun < p_best_fitness[g_best_idx]:
                    g_best = result.x
                    g_best_idx = np.argmin(p_best_fitness)
                    p_best[g_best_idx] = result.x
                    p_best_fitness[g_best_idx] = result.fun

        return g_best

    def initialize_population(self, size):
        return np.random.uniform(self.lb, self.ub, (size, self.dim))

    def adaptive_enforce_periodicity(self, vector):
        if self.dim % 2 == 0:
            period = 2
        else:
            period = 3
        num_periods = self.dim // period
        for i in range(num_periods):
            mean_value = np.mean(vector[i*period:(i+1)*period])
            vector[i*period:(i+1)*period] = mean_value
        return vector