import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def differential_evolution(self, func, bounds, population_size=20, generations=100):
        pop = np.random.rand(population_size, self.dim)
        pop = bounds[:, 0] + pop * (bounds[:, 1] - bounds[:, 0])
        fitness = np.array([func(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        best = pop[best_idx]

        for gen in range(generations):
            if self.budget <= 0:  # Dynamic budget check
                break
            for i in range(population_size):
                idxs = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + 0.8 * (b - c), bounds[:, 0], bounds[:, 1])
                cross_points = np.random.rand(self.dim) < 0.9
                trial = np.where(cross_points, mutant, pop[i])
                f_trial = func(trial)
                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    pop[i] = trial
                    if f_trial < fitness[best_idx]:
                        best_idx = i
                        best = trial
                self.budget -= 1  # Decrement budget
            if gen % 10 == 0 and gen * population_size >= self.budget:
                break

        return best

    def nelder_mead(self, func, x0):
        result = minimize(func, x0, method='Nelder-Mead', options={'maxfev': max(1, self.budget // 2)})
        return result.x, result.fun

    def __call__(self, func):
        layer_increment = max(1, self.dim // 10)
        current_dim = layer_increment
        best_solution = None
        best_fitness = float('inf')

        while current_dim <= self.dim and self.budget > 0:
            subset_func = lambda x: func(np.pad(x, (0, self.dim - len(x)), 'constant'))
            bounds = np.array([[func.bounds.lb[i], func.bounds.ub[i]] for i in range(current_dim)])
            de_solution = self.differential_evolution(subset_func, bounds, population_size=20, generations=100)
            
            local_solution, local_fitness = self.nelder_mead(subset_func, de_solution)
            if local_fitness < best_fitness:
                best_fitness = local_fitness
                best_solution = np.pad(local_solution, (0, self.dim - len(local_solution)), 'constant')

            current_dim += layer_increment
        
        return best_solution