import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

class HybridOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _initialize_population(self, size, bounds):
        lb, ub = bounds.lb, bounds.ub
        # Enhanced initialization using Sobol sequences for better coverage
        sampler = Sobol(d=self.dim, scramble=True)
        pop = lb + (ub - lb) * sampler.random(size)
        pop[:size // 2] = lb + (ub - pop[size // 2:])
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

    def _local_search(self, solution, func, bounds):
        res = minimize(func, solution, method='L-BFGS-B', bounds=np.vstack((bounds.lb, bounds.ub)).T)
        return res.x

    def _periodic_cost(self, solution):
        period_length = self.dim // 2
        periodic_part = solution[:period_length]
        periodicity_penalty = np.sum((solution - np.tile(periodic_part, self.dim // period_length))**2)
        return periodicity_penalty

    def __call__(self, func):
        bounds = func.bounds
        population_size = 20
        num_generations = self.budget // population_size
        pop = self._initialize_population(population_size, bounds)

        for gen in range(num_generations):
            pop = self._differential_evolution(pop, lambda x: func(x) + self._periodic_cost(x), bounds)
            if gen % 5 == 0:  # Adaptive local search invocation
                best_idx = np.argmin([func(ind) for ind in pop])
                best_solution = pop[best_idx]
                refined_solution = self._local_search(best_solution, func, bounds)
                worst_idx = np.argmax([func(ind) for ind in pop])
                pop[worst_idx] = refined_solution

        best_idx = np.argmin([func(ind) for ind in pop])
        return pop[best_idx]