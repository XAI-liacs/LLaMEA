import numpy as np
from scipy.optimize import minimize

class HybridPSONM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.inertia = 0.6
        self.cognitive_coef = 1.2
        self.social_coef = 1.2
        self.init_budget = budget // 2
        self.local_budget = budget - self.init_budget
        self.velocities = np.zeros((self.population_size, self.dim))

    def _initialize_population(self, bounds):
        lb, ub = bounds
        return np.random.uniform(lb, ub, (self.population_size, self.dim)) * 0.8

    def _update_velocities(self, population, personal_best, global_best):
        r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
        self.velocities = (self.inertia * self.velocities + 
                           self.cognitive_coef * r1 * (personal_best - population) +
                           self.social_coef * r2 * (global_best - population))

    def _pso_step(self, population, func, bounds):
        lb, ub = bounds
        personal_best = np.copy(population)
        personal_best_scores = np.array([func(ind) for ind in personal_best])
        global_best = personal_best[np.argmin(personal_best_scores)]

        new_population = np.copy(population)
        for _ in range(self.init_budget // self.population_size):
            self._update_velocities(population, personal_best, global_best)
            new_population = population + self.velocities
            new_population = np.clip(new_population, lb, ub)
            scores = np.array([func(ind) for ind in new_population])
            better_mask = scores < personal_best_scores
            personal_best_scores[better_mask] = scores[better_mask]
            personal_best[better_mask] = new_population[better_mask]
            if np.min(scores) < func(global_best):
                global_best = new_population[np.argmin(scores)]
            population = new_population
        return global_best

    def _encourage_periodicity(self, solution, bounds):
        lb, ub = bounds
        period = max(2, self.dim // 4)
        periodic_solution = np.tile(solution[:period], self.dim // period + 1)[:self.dim]
        return np.clip(periodic_solution, lb, ub)

    def _local_refinement(self, solution, func, bounds):
        perturbed_solution = self._encourage_periodicity(solution, bounds)  # Change 1: Encourage periodicity initially
        result = minimize(func, perturbed_solution, method='Nelder-Mead', options={'maxfev': self.local_budget})
        return result.x

    def _periodicity_aware_perturbation(self, solution, bounds):  # Change 2: New function
        period = self.dim // 4  # Change 3: Adjust periodic perturbation
        perturbation = np.sin(np.linspace(0, 2 * np.pi, period))  # Change 4: Use sinusoidal perturbation
        perturbed_solution = solution + np.tile(perturbation, self.dim // period)
        return np.clip(perturbed_solution, bounds[0], bounds[1])

    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        population = self._initialize_population(bounds)

        best_solution = self._pso_step(population, func, bounds)
        best_solution = self._periodicity_aware_perturbation(best_solution, bounds)  # Change 5: Apply perturbation

        if self.local_budget > 0:
            best_solution = self._local_refinement(best_solution, func, bounds)

        return best_solution