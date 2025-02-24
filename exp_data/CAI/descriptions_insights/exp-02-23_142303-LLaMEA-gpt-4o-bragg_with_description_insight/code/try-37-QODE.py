import numpy as np
from scipy.optimize import minimize

class QODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.init_budget = budget // 2
        self.local_budget = budget - self.init_budget
        self.F = 0.8
        self.CR = 0.9
        self.velocities = np.zeros((self.population_size, self.dim))

    def _initialize_population(self, bounds):
        lb, ub = bounds
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        opposite_population = lb + ub - population
        combined_population = np.vstack((population, opposite_population))
        return combined_population[:self.population_size]

    def _mutate(self, population):
        idxs = np.random.choice(self.population_size, 3, replace=False)
        a, b, c = population[idxs]
        mutant_vector = a + self.F * (b - c)
        return mutant_vector

    def _crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.CR
        trial_vector = np.where(crossover_mask, mutant, target)
        return trial_vector

    def _encourage_periodicity(self, solution, bounds):
        lb, ub = bounds
        period = max(2, self.dim // 4)
        periodic_solution = np.tile(solution[:period], self.dim // period + 1)[:self.dim]
        return np.clip(periodic_solution, lb, ub)

    def _local_refinement(self, solution, func, bounds):
        result = minimize(func, solution, method='Nelder-Mead', options={'maxfev': self.local_budget})
        return result.x

    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        population = self._initialize_population(bounds)
        best_solution = None
        best_score = float('inf')

        # Differential Evolution
        for _ in range(self.init_budget // self.population_size):
            for i in range(self.population_size):
                target = population[i]
                mutant = self._mutate(population)
                trial = self._crossover(target, mutant)
                trial = np.clip(trial, *bounds)
                trial_score = func(trial)
                target_score = func(target)
                if trial_score < target_score:
                    population[i] = trial
                    if trial_score < best_score:
                        best_score = trial_score
                        best_solution = trial

        # Encourage periodicity
        best_solution = self._encourage_periodicity(best_solution, bounds)
        
        # Local refinement
        if self.local_budget > 0:
            best_solution = self._local_refinement(best_solution, func, bounds)

        return best_solution