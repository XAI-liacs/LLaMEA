import numpy as np
from scipy.optimize import minimize

class HybridDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(15 * dim, budget // 10)
        self.f = 0.9
        self.cr = 0.9
        self.current_evals = 0

    def initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        return pop

    def cooperative_coevolution(self, func, population, bounds):
        subcomponents = np.array_split(np.arange(self.dim), 4)  # Decompose into 4 subcomponents
        for subcomponent in subcomponents:
            for i in range(self.population_size):
                trial = np.copy(population[i])
                if self.current_evals >= self.budget:
                    break
                idxs = list(range(self.population_size))
                idxs.remove(i)
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                donor = a[subcomponent] + self.f * (b[subcomponent] - c[subcomponent])
                donor = np.clip(donor, bounds.lb[subcomponent], bounds.ub[subcomponent])
                for j in subcomponent:
                    if np.random.rand() < self.cr:
                        trial[j] = donor[j]
                trial = self.ensure_periodicity(trial, period=2)
                if func(trial) < func(population[i]):
                    population[i] = trial
                self.current_evals += 1
        return population

    def ensure_periodicity(self, solution, period):
        return np.tile(solution[:period], self.dim // period + 1)[:self.dim]

    def differential_evolution(self, func, bounds):
        population = self.initialize_population(bounds)
        best_solution = None
        best_score = float('inf')

        while self.current_evals < self.budget:
            population = self.cooperative_coevolution(func, population, bounds)
            for sol in population:
                score = func(sol)
                if score < best_score:
                    best_score = score
                    best_solution = sol
        return best_solution

    def local_refinement(self, func, best_solution, bounds):
        result = minimize(func, best_solution, bounds=[(bounds.lb[i], bounds.ub[i]) for i in range(self.dim)], method='L-BFGS-B')
        return result.x

    def __call__(self, func):
        bounds = func.bounds
        initial_solution = self.differential_evolution(func, bounds)
        refined_solution = self.local_refinement(func, initial_solution, bounds)
        return refined_solution