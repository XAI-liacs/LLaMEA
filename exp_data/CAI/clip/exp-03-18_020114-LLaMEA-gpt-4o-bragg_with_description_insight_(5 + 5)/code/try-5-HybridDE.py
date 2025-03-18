import numpy as np
from scipy.optimize import minimize

class HybridDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.func_evals = 0

    def periodic_restriction(self, x):
        """Encourage periodic solutions by penalizing deviations from periodicity."""
        period = 2
        penalties = np.sum((x - np.roll(x, period))**2) * 0.1  # Adjust penalty weight

    def differential_evolution(self, func, bounds):
        lb, ub = bounds.lb, bounds.ub
        population = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        best_solution = None
        best_score = float('inf')

        while self.func_evals < self.budget:
            new_population = np.copy(population)
            for i in range(self.population_size):
                idxs = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = population[idxs]
                mutant = np.clip(x0 + self.F * (x1 - x2), lb, ub)
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                # Add periodic penalty
                score = func(trial) + self.periodic_restriction(trial)
                self.func_evals += 1

                if score < func(population[i]):
                    new_population[i] = trial
                    if score < best_score:
                        best_score = score
                        best_solution = trial
                
                # Check budget
                if self.func_evals >= self.budget:
                    break

            population = new_population
        
        return best_solution

    def local_optimization(self, func, x0, bounds):
        result = minimize(func, x0, bounds=bounds, method='L-BFGS-B')
        return result.x if result.success else x0

    def __call__(self, func):
        bounds = func.bounds
        solution = self.differential_evolution(func, bounds)
        solution = self.local_optimization(func, solution, bounds)
        return solution