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
            CR = 0.5 + np.random.rand() * 0.3  # Change 2: Narrower CR for stability
            pop_size = int(np.clip(self.budget / (10 * self.dim), 5, 50))  # Change: Dynamic pop_size
            for i in range(pop_size):
                if self.eval_count >= self.budget:
                    break
                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                # Change 3: Introduce layer-wise perturbation
                layer_perturb = np.random.rand(self.dim) < (1.0 / self.dim)
                mutant = np.clip(a + F * (b - c) * layer_perturb, bounds.lb, bounds.ub)
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
        # Change 4: Initialize local search with perturbed best solution
        perturbation = np.random.normal(0, 0.01, size=self.dim)
        init_x0 = np.clip(x0 + perturbation, bounds.lb, bounds.ub)
        result = minimize(func, init_x0, bounds=[(bounds.lb[i], bounds.ub[i]) for i in range(self.dim)], 
                          method='L-BFGS-B', options={'maxiter': 120, 'ftol': ftol})
        return result.x

    def __call__(self, func):
        bounds = func.bounds
        best_solution = self.differential_evolution(func, bounds)
        final_solution = self.local_search(func, best_solution, bounds)
        return final_solution