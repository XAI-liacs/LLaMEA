import numpy as np
from scipy.optimize import minimize

class HybridDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20  # Population size for DE
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.bounds = None
        self.eval_count = 0

    def quasi_oppositional_init(self, lb, ub):
        midpoint = (lb + ub) / 2
        pop = lb + np.random.rand(self.pop_size, self.dim) * (ub - lb)
        opp_pop = midpoint + (midpoint - pop)
        combined_pop = np.concatenate((pop, opp_pop), axis=0)
        return combined_pop[:self.pop_size]

    def differential_evolution(self, pop, func):
        new_pop = np.empty_like(pop)
        for i in range(self.pop_size):
            if self.eval_count >= self.budget:
                break
            indices = np.random.choice(self.pop_size, 3, replace=False)
            x0, x1, x2 = pop[indices]
            mutant = np.clip(x0 + self.F * (x1 - x2), self.bounds.lb, self.bounds.ub)
            cross_points = np.random.rand(self.dim) < self.CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, pop[i])
            trial_fitness = func(trial)
            self.eval_count += 1
            if trial_fitness < func(pop[i]):
                new_pop[i] = trial
            else:
                new_pop[i] = pop[i]
        return new_pop

    def local_search(self, best_solution, func):
        if self.eval_count >= self.budget:
            return best_solution
        result = minimize(func, best_solution, bounds=list(zip(self.bounds.lb, self.bounds.ub)), method='L-BFGS-B')
        self.eval_count += result.nfev
        return result.x

    def __call__(self, func):
        self.bounds = func.bounds
        pop = self.quasi_oppositional_init(self.bounds.lb, self.bounds.ub)
        best_solution = pop[np.argmin([func(ind) for ind in pop])]
        self.eval_count += self.pop_size

        while self.eval_count < self.budget:
            pop = self.differential_evolution(pop, func)
            current_best = pop[np.argmin([func(ind) for ind in pop])]
            self.eval_count += self.pop_size

            # Periodically apply local search to refine the best solution
            if self.eval_count < self.budget:
                best_solution = self.local_search(current_best, func)
        
        return best_solution