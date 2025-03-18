import numpy as np
from scipy.optimize import minimize

class HybridDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(20 * dim, budget // 8)  # Adjusted population size
        self.f = 0.6 + np.random.rand() * 0.2  # Adjusted scaling factor range
        self.cr = 0.6 + np.random.rand() * 0.3  # Adjusted crossover rate range
        self.current_evals = 0

    def initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        center = (lb + ub) / 2
        radius = (ub - lb) / 3  # Increased search radius
        pop = np.random.uniform(center - radius, center + radius, (self.population_size, self.dim))
        return pop

    def ensure_periodicity(self, solution, period):
        return np.tile(solution[:period], self.dim // period + 1)[:self.dim]

    def differential_evolution(self, func, bounds):
        population = self.initialize_population(bounds)
        best_solution = None
        best_score = float('inf')
        dynamic_noise = 0.3  # Introduced dynamic noise factor
        
        while self.current_evals < self.budget:
            for i in range(self.population_size):
                if self.current_evals >= self.budget:
                    break
                
                idxs = list(range(self.population_size))
                idxs.remove(i)
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                dynamic_f = self.f * (1 + 0.1 * np.sin(2 * np.pi * self.current_evals / self.budget))  # Added sinusoidal variation
                donor = a + dynamic_f * (b - c) + np.random.normal(0, dynamic_noise, self.dim)  # Applied dynamic noise
                donor = np.clip(donor, bounds.lb, bounds.ub)

                trial = np.copy(population[i])
                self.cr = 0.9 + 0.1 * np.cos(2 * np.pi * self.current_evals / self.budget)  # Alternating crossover rate
                for j in range(self.dim):
                    if np.random.rand() < self.cr:
                        trial[j] = donor[j]
                
                trial = self.ensure_periodicity(trial, period=2)

                score = func(trial)
                if score < best_score:
                    best_score = score
                    best_solution = trial
                
                if score < func(population[i]):
                    population[i] = trial

                self.current_evals += 1

        return best_solution

    def local_refinement(self, func, best_solution, bounds):
        if np.random.rand() < 0.5:
            result = minimize(func, best_solution, bounds=[(bounds.lb[i], bounds.ub[i]) for i in range(self.dim)], method='L-BFGS-B')
            return result.x
        return best_solution

    def __call__(self, func):
        bounds = func.bounds
        initial_solution = self.differential_evolution(func, bounds)
        refined_solution = self.local_refinement(func, initial_solution, bounds)
        return refined_solution