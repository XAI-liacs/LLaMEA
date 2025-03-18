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
            F_dynamic = 0.5 + np.random.rand() * 0.5  # Change 1: Wider F range for exploration
            CR = 0.5 + np.random.rand() * 0.4  # Change 2: Wider CR range for more diversity
            pop_size = int(np.clip(self.budget / (8 * self.dim), 5, 50))  # Change 3: Adjusted pop_size calculation
            for i in range(pop_size):
                if self.eval_count >= self.budget:
                    break
                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + F_dynamic * (b - c), bounds.lb, bounds.ub)
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
        result = minimize(func, x0, bounds=[(max(bounds.lb[i], x0[i] - 0.05), min(bounds.ub[i], x0[i] + 0.05)) for i in range(self.dim)],  # Change 4: Narrower search bounds
                          method='L-BFGS-B', options={'maxiter': 150, 'ftol': 1e-5}, x0=x0)  # Change 5 & 6: Increased maxiter and refined ftol further
        return result.x

    def __call__(self, func):
        bounds = func.bounds
        best_solution = self.differential_evolution(func, bounds)
        final_solution = self.local_search(func, best_solution, bounds)
        return final_solution