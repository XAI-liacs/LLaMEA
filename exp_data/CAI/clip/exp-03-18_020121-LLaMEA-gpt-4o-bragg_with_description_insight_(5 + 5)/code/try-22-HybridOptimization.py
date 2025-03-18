import numpy as np
from scipy.optimize import minimize

class HybridOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _initialize_population(self, size, bounds):
        lb, ub = bounds.lb, bounds.ub
        pop = np.random.uniform(lb, ub, (size, self.dim))
        # Symmetric initialization to enhance exploration
        pop[:size//2] = lb + (ub - pop[size//2:])
        return pop

    def _differential_evolution(self, pop, func, bounds, F=0.8, CR=0.9):
        size = len(pop)
        elite = pop[np.argmin([func(ind) for ind in pop])]  # Elitism: retain the best individual
        for i in range(size):
            # Randomly select three distinct vectors
            indices = np.random.choice(np.delete(np.arange(size), i), 3, replace=False)
            a, b, c = pop[indices]
            # Mutation
            F_adaptive = 0.5 + 0.5 * np.random.rand()  # Adaptive mutation factor
            mutant = np.clip(a + F_adaptive * (b - c), bounds.lb, bounds.ub)
            # Recombination
            cross_points = np.random.rand(self.dim) < CR
            trial = np.where(cross_points, mutant, pop[i])
            # Selection
            if func(trial) < func(pop[i]):
                pop[i] = trial
        pop[np.argmax([func(ind) for ind in pop])] = elite  # Incorporate elite individual back into population
        return pop

    def _local_search(self, solution, func, bounds):
        # Local refinement using BFGS
        res = minimize(func, solution, method='L-BFGS-B', bounds=np.vstack((bounds.lb, bounds.ub)).T)
        return res.x

    def _periodic_cost(self, solution):
        # Custom cost function to encourage periodicity
        period_length = self.dim // 2
        periodic_part = solution[:period_length]
        periodicity_penalty = np.sum((solution - np.tile(periodic_part, self.dim // period_length))**2)
        return periodicity_penalty

    def __call__(self, func):
        bounds = func.bounds
        population_size = 20
        num_generations = self.budget // population_size
        pop = self._initialize_population(population_size, bounds)

        for _ in range(num_generations):
            pop = self._differential_evolution(pop, lambda x: func(x) + self._periodic_cost(x), bounds)
            best_idx = np.argmin([func(ind) for ind in pop])
            best_solution = pop[best_idx]
            refined_solution = self._local_search(best_solution, func, bounds)
            # Replace worst with refined solution to maintain diversity
            worst_idx = np.argmax([func(ind) for ind in pop])
            pop[worst_idx] = refined_solution

        # Return the best found solution
        best_idx = np.argmin([func(ind) for ind in pop])
        return pop[best_idx]