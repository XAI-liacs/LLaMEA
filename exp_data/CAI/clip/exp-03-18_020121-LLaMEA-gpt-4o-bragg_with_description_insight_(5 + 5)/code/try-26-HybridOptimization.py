import numpy as np
from scipy.optimize import minimize

class HybridOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _initialize_population(self, size, bounds):
        lb, ub = bounds.lb, bounds.ub
        pop = np.random.uniform(lb, ub, (size, self.dim))
        pop[:size//2] = lb + (ub - pop[size//2:])
        return pop

    def _differential_evolution(self, pop, func, bounds, F=0.8, CR=0.9):
        size = len(pop)
        for i in range(size):
            indices = np.random.choice(np.delete(np.arange(size), i), 3, replace=False)
            a, b, c = pop[indices]
            mutant = np.clip(a + F * (b - c), bounds.lb, bounds.ub)
            cross_points = np.random.rand(self.dim) < CR
            trial = np.where(cross_points, mutant, pop[i])
            if func(trial) < func(pop[i]):
                pop[i] = trial
        return pop

    def _adaptive_local_search(self, solution, func, bounds):
        res = minimize(func, solution, method='L-BFGS-B', bounds=np.vstack((bounds.lb, bounds.ub)).T)
        return res.x

    def _adaptive_periodic_cost(self, solution):
        adaptive_period_length = max(1, self.dim // 4)
        periodic_part = solution[:adaptive_period_length]
        periodicity_penalty = np.sum((solution - np.tile(periodic_part, self.dim // adaptive_period_length))**2)
        return periodicity_penalty

    def __call__(self, func):
        bounds = func.bounds
        population_size = 20
        num_generations = self.budget // population_size
        pop = self._initialize_population(population_size, bounds)

        for gen in range(num_generations):
            # Adjust the mutation factor F based on the generation number
            F = 0.9 - 0.5 * (gen / num_generations)
            pop = self._differential_evolution(pop, lambda x: func(x) + self._adaptive_periodic_cost(x), bounds, F=F)
            best_idx = np.argmin([func(ind) for ind in pop])
            best_solution = pop[best_idx]
            refined_solution = self._adaptive_local_search(best_solution, func, bounds)
            worst_idx = np.argmax([func(ind) for ind in pop])
            pop[worst_idx] = refined_solution

        best_idx = np.argmin([func(ind) for ind in pop])
        return pop[best_idx]