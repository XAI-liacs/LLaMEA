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

    def _adaptive_mutation_factor(self, generation, max_generations):
        return 0.7 + (0.9 - 0.7) * (1 - generation / max_generations)  # Adjusted mutation factor range

    def _crowding_distance(self, pop, func):
        fitness = np.array([func(ind) for ind in pop])
        order = np.argsort(fitness)
        distances = np.zeros(len(pop))
        for i in range(1, len(pop) - 1):
            distances[order[i]] = fitness[order[i + 1]] - fitness[order[i - 1]]
        return distances

    def _differential_evolution(self, pop, func, bounds, generation, max_generations):
        size = len(pop)
        F = self._adaptive_mutation_factor(generation, max_generations)
        CR = 0.8  # Reduced crossover rate
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
        best_solution = None

        for generation in range(num_generations):
            pop = self._differential_evolution(pop, lambda x: func(x) + self._periodic_cost(x), bounds, generation, num_generations)
            current_best_idx = np.argmin([func(ind) for ind in pop])
            current_best_solution = pop[current_best_idx]
            if best_solution is None or func(current_best_solution) < func(best_solution):
                best_solution = current_best_solution  # Retain elitism
            refined_solution = self._local_search(current_best_solution, func, bounds)
            worst_idx = np.argmax(self._crowding_distance(pop, func))
            pop[worst_idx] = refined_solution

        return best_solution