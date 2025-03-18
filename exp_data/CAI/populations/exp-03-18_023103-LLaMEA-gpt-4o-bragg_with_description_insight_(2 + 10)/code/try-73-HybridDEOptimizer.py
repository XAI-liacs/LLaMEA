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
            self.F = np.random.uniform(0.5, 0.9)  # Adapted F range for diversity
            mutant = np.clip(x0 + self.F * (x1 - x2), self.bounds.lb, self.bounds.ub)
            adaptive_CR = np.random.uniform(0.8, 1.0)
            cross_points = np.random.rand(self.dim) < adaptive_CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, pop[i])
            trial_fitness = func(trial)
            self.eval_count += 1
            if trial_fitness < func(pop[i]):
                # Enforce pattern-based periodicity
                new_pop[i] = self.enforce_periodicity(trial, pattern_size=4)
            else:
                new_pop[i] = pop[i]
        return new_pop

    def enforce_periodicity(self, solution, pattern_size=4):
        # Use smaller repeating patterns for periodicity
        for i in range(0, self.dim, pattern_size):
            pattern = solution[i:i+pattern_size]
            for j in range(i, self.dim, pattern_size):
                solution[j:j+pattern_size] = pattern
        return solution

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

            if self.eval_count < self.budget and self.eval_count % (self.pop_size // 2) == 0:
                best_solution = self.local_search(current_best, func)
        
        return best_solution