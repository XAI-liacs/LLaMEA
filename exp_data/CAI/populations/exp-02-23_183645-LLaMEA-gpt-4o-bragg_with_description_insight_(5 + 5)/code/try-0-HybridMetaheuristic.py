import numpy as np
from scipy.optimize import minimize

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim  # Typical rule of thumb for DE
        self.f = 0.8  # DE mutation factor
        self.cr = 0.7  # DE crossover probability
    
    def __call__(self, func):
        np.random.seed(42)
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        best_solution = None
        best_score = float('inf')
        func_calls = 0

        while func_calls < self.budget:
            new_population = []
            for i in range(self.population_size):
                if func_calls >= self.budget:
                    break
                # Differential Evolution Mutation
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.f * (b - c), lb, ub)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.cr
                trial = np.where(crossover_mask, mutant, population[i])

                # Evaluate
                trial_score = func(trial)
                func_calls += 1

                # Select
                if trial_score < func(population[i]):
                    new_population.append(trial)
                    if trial_score < best_score:
                        best_solution, best_score = trial, trial_score
                else:
                    new_population.append(population[i])
            
            population = np.array(new_population)

            # Periodicity encouragement
            for i in range(self.population_size):
                if func_calls >= self.budget:
                    break
                trial = self._encourage_periodicity(population[i])
                trial_score = func(trial)
                func_calls += 1
                if trial_score < best_score:
                    best_solution, best_score = trial, trial_score

            # Local Optimization with BFGS
            if func_calls + self.dim <= self.budget:
                result = minimize(func, best_solution, method='L-BFGS-B', bounds=[(lb, ub)]*self.dim)
                func_calls += result.nfev
                if result.fun < best_score:
                    best_solution, best_score = result.x, result.fun

        return best_solution, best_score

    def _encourage_periodicity(self, solution):
        # Encourage periodicity by averaging segments of the solution
        mid = self.dim // 2
        periodic_solution = np.concatenate([solution[:mid], solution[:mid]])
        return np.clip(periodic_solution, solution.min(), solution.max())