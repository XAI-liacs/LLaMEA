import numpy as np
from scipy.optimize import minimize

class HybridDEBFGS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def _de_step(self, pop, bounds, F=0.8, CR=0.9):
        # Differential Evolution step
        new_pop = np.copy(pop)
        for i in range(len(pop)):
            indices = np.random.choice(len(pop), 3, replace=False)
            a, b, c = pop[indices]
            mutant = np.clip(a + F * (b - c), bounds.lb, bounds.ub)
            cross_points = np.random.rand(self.dim) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, pop[i])
            new_pop[i] = trial
        return new_pop

    def _encourage_periodicity(self, solution):
        # Encourage periodicity by averaging over blocks
        for i in range(0, len(solution), 2):
            solution[i:(i+2)] = np.mean(solution[i:(i+2)])
        return solution

    def _local_search(self, x, func, bounds):
        # BFGS local optimization
        result = minimize(func, x, bounds=[(lb, ub) for lb, ub in zip(bounds.lb, bounds.ub)], method='L-BFGS-B')
        return result.x

    def __call__(self, func):
        # Initialize population uniformly within bounds
        bounds = func.bounds
        pop_size = 10 * self.dim
        pop = np.random.uniform(bounds.lb, bounds.ub, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        self.evaluations += pop_size

        while self.evaluations < self.budget:
            new_pop = self._de_step(pop, bounds)
            for i in range(len(new_pop)):
                new_pop[i] = self._encourage_periodicity(new_pop[i])
                if self.evaluations < self.budget:
                    trial_fitness = func(new_pop[i])
                    self.evaluations += 1
                    if trial_fitness < fitness[i]:
                        fitness[i] = trial_fitness
                        pop[i] = new_pop[i]

            # Local search on the best individual
            if self.evaluations < self.budget:
                best_idx = np.argmin(fitness)
                pop[best_idx] = self._local_search(pop[best_idx], func, bounds)
                fitness[best_idx] = func(pop[best_idx])
                self.evaluations += 1

        best_idx = np.argmin(fitness)
        return pop[best_idx]