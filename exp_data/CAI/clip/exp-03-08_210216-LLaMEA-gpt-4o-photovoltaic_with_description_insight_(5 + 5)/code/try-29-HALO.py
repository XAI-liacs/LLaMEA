import numpy as np
from scipy.optimize import minimize

class HALO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.F = 0.8  # Differential evolution scale factor
        self.CR = 0.5  # Changed from 0.9 to 0.5 for adaptive crossover probability
        self.pop = None
        self.func_evals = 0

    def _initialize_population(self, lb, ub):
        self.pop = np.random.uniform(lb, ub, (self.population_size, self.dim))

    def _evaluate_population(self, func):
        fitness = np.array([func(ind) for ind in self.pop])
        self.func_evals += self.population_size
        return fitness

    def _differential_evolution_step(self, fitness, lb, ub, func):
        next_pop = np.empty_like(self.pop)
        for i in range(self.population_size):
            indices = np.random.choice(self.population_size, 3, replace=False)
            a, b, c = self.pop[indices]
            self.F = 0.5 + 0.3 * (1 - self.func_evals / self.budget)  # Adaptive mutation scaling
            mutant = np.clip(a + self.F * (b - c), lb, ub)
            cross_points = np.random.rand(self.dim) < self.CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, self.pop[i])
            trial_fitness = func(trial)
            self.func_evals += 1
            if trial_fitness < fitness[i]:
                next_pop[i] = trial
            else:
                next_pop[i] = self.pop[i]
        self.pop = next_pop

    def _local_refinement(self, func, lb, ub):
        if self.func_evals < (3 * self.budget) // 4:  # Adjusted local refinement frequency
            best_idx = np.argmin(self._evaluate_population(func))
            best_solution = self.pop[best_idx]
            result = minimize(func, best_solution, bounds=[(lb[i], ub[i]) for i in range(self.dim)], method='L-BFGS-B')
            if result.success and func(result.x) < func(best_solution):
                self.pop[best_idx] = result.x
                self.func_evals += 1

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self._initialize_population(lb, ub)
        fitness = self._evaluate_population(func)

        while self.func_evals < self.budget:
            self._differential_evolution_step(fitness, lb, ub, func)
            fitness = self._evaluate_population(func)

        best_idx = np.argmin(fitness)
        return self.pop[best_idx]