import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.eval_count = 0

    def differential_evolution(self, func, bounds, pop_size=20, F=0.8, CR=0.9):
        pop = np.random.uniform(bounds.lb, bounds.ub, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        best = pop[best_idx]
        
        while self.eval_count < self.budget:
            F = 0.5 + np.random.rand() * 0.3  # Change 1: Narrower F for stability
            fitness_diversity = np.std(fitness) / np.mean(fitness)  # New Change: Fitness diversity metric
            CR = 0.5 + fitness_diversity * 0.4  # New Change: Adaptive CR based on fitness diversity
            pop_size = int(np.clip(self.budget / (10 * self.dim), 5, 50))  # Change: Dynamic pop_size
            for i in range(pop_size):
                if self.eval_count >= self.budget:
                    break
                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), bounds.lb, bounds.ub)
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                trial_fitness = func(trial)
                self.eval_count += 1
                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    pop[i] = trial
                    if trial_fitness < fitness[best_idx]:
                        best_idx = i
                        best = trial
        return best

    def local_search(self, func, x0, bounds):
        ftol = 1e-6 * (1 + (self.budget - self.eval_count) / self.budget)  # Change: Adaptive ftol
        random_start = np.random.uniform(bounds.lb, bounds.ub)  # New Change: Random start point
        result = minimize(func, random_start, bounds=[(bounds.lb[i], bounds.ub[i]) for i in range(self.dim)], 
                          method='L-BFGS-B', options={'maxiter': 120, 'ftol': ftol}, x0=x0)
        return result.x

    def __call__(self, func):
        bounds = func.bounds
        best_solution = self.differential_evolution(func, bounds)
        final_solution = self.local_search(func, best_solution, bounds)
        return final_solution