import numpy as np
from scipy.optimize import minimize

class PeriodicDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_factor = 0.8
        self.crossover_probability = 0.7
        self.periodicity_factor = 0.9
        self.evaluations = 0

    def initialize_population(self, lb, ub):
        return np.random.uniform(lb, ub, (self.population_size, self.dim))

    def mutate(self, population):
        indices = np.random.choice(range(self.population_size), 3, replace=False)
        a, b, c = population[indices]
        self.mutation_factor = 0.6 + 0.2 * np.random.rand()  # Improved adaptive mutation factor
        return a + self.mutation_factor * (b - c)

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.crossover_probability
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def periodic_correction(self, candidate, lb, ub):
        midpoint = (ub + lb) / 2
        half_range = (ub - lb) / 2
        return midpoint + self.periodicity_factor * (candidate - midpoint) % half_range

    def local_optimize(self, candidate, func):
        result = minimize(func, candidate, method='L-BFGS-B', options={'maxiter': 10}, bounds=list(zip(func.bounds.lb, func.bounds.ub)))  # Refined local optimization
        return result.x if result.success else candidate

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = self.initialize_population(lb, ub)
        best_candidate = None
        best_value = np.inf

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break
                
                target = population[i]
                mutant = self.mutate(population)
                trial = self.crossover(target, mutant)
                trial = self.periodic_correction(trial, lb, ub)

                f_target = func(target)
                f_trial = func(trial)
                self.evaluations += 2

                if f_trial < f_target:
                    population[i] = trial
                    if f_trial < best_value:
                        best_candidate, best_value = trial, f_trial

            if best_candidate is not None:
                best_candidate = self.local_optimize(best_candidate, func)
                if func(best_candidate) < best_value:
                    best_value = func(best_candidate)
                self.evaluations += 1

        return best_candidate