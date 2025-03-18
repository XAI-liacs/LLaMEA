import numpy as np
from scipy.optimize import minimize

class HybridDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(20 * dim, budget // 10)  # Increased population size for broader exploration
        self.f = 0.9  # Enhanced differential weight for improved exploration
        self.cr = 0.9  # Increased crossover probability for better diversity
        self.current_evals = 0

    def initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))  # Full-range initialization for diversity
        return pop

    def ensure_periodicity(self, solution, period):
        return np.tile(solution[:period], self.dim // period + 1)[:self.dim]

    def differential_evolution(self, func, bounds):
        population = self.initialize_population(bounds)
        best_solution = None
        best_score = float('inf')

        while self.current_evals < self.budget:
            for i in range(self.population_size):
                if self.current_evals >= self.budget:
                    break
                
                idxs = list(range(self.population_size))
                idxs.remove(i)
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                donor = a + self.f * (b - c)
                donor = np.clip(donor, bounds.lb, bounds.ub)

                trial = np.copy(population[i])
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
        result = minimize(func, best_solution, bounds=[(bounds.lb[i], bounds.ub[i]) for i in range(self.dim)], method='L-BFGS-B')
        return result.x

    def __call__(self, func):
        bounds = func.bounds
        initial_solution = self.differential_evolution(func, bounds)
        refined_solution = self.local_refinement(func, initial_solution, bounds)
        return refined_solution