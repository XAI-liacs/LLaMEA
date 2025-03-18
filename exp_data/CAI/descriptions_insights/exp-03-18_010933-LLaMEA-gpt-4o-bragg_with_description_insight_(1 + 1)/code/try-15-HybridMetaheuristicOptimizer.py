import numpy as np
from scipy.optimize import minimize

class HybridMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.current_evaluations = 0

    def differential_evolution(self, func, bounds, pop_size=20, F=0.8, CR=0.95):
        pop = np.random.uniform(bounds.lb, bounds.ub, (pop_size, self.dim))
        opp_pop = bounds.ub + bounds.lb - pop
        pop = np.concatenate((pop, opp_pop))
        best_idx = np.argmin([func(ind) for ind in pop])
        best = pop[best_idx]
        self.current_evaluations += pop_size

        while self.current_evaluations < self.budget:
            for i in range(pop_size):
                if self.current_evaluations >= self.budget:
                    break

                indices = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                F = 0.5 + 0.3 * np.random.rand()
                CR = 0.9 + 0.1 * np.random.rand()
                mutant = np.clip(a + F * (b - c), bounds.lb, bounds.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])
                
                # Enforcing periodicity in potential solutions
                trial = trial - np.mod(trial, 2 * (bounds.ub[0] - bounds.lb[0]) / self.dim)

                trial_fitness = func(trial)
                self.current_evaluations += 1

                # Dynamically adjust trial acceptance criteria
                if trial_fitness < func(pop[i]) or np.random.rand() < 0.1:
                    pop[i] = trial
                    if trial_fitness < func(best):
                        best = trial

            pop_size = max(pop_size // 2, 10)

        return best

    def local_search(self, func, x0, bounds):
        result = minimize(func, x0, bounds=bounds, method='L-BFGS-B')
        return result.x

    def __call__(self, func):
        bounds = list(zip(func.bounds.lb, func.bounds.ub))
        best_de = self.differential_evolution(func, func.bounds)

        if self.current_evaluations < self.budget:
            best_solution = self.local_search(func, best_de, bounds)
        else:
            best_solution = best_de

        return best_solution