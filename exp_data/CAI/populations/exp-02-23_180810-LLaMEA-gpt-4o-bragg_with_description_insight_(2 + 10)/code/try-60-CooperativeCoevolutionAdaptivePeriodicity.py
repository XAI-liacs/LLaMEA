import numpy as np
from scipy.optimize import minimize

class CooperativeCoevolutionAdaptivePeriodicity:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7
        self.evaluations = 0
        self.subcomponent_size = int(np.ceil(self.dim / 2))

    def initialize_population(self, bounds):
        lb, ub = bounds
        population = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        return population

    def cooperative_coevolution(self, func, bounds):
        population = self.initialize_population(bounds)
        best_solution = None
        best_score = float('-inf')
        
        while self.evaluations < self.budget:
            for start in range(0, self.dim, self.subcomponent_size):
                subcomponent_bounds = (bounds[0][start:start + self.subcomponent_size], 
                                       bounds[1][start:start + self.subcomponent_size])
                
                best_subcomponent_solution = np.random.rand(self.subcomponent_size) * (subcomponent_bounds[1] - subcomponent_bounds[0]) + subcomponent_bounds[0]
                for i in range(self.population_size):
                    if self.evaluations >= self.budget:
                        break

                    idxs = np.random.choice(self.population_size, 3, replace=False)
                    a, b, c = population[idxs, start:start + self.subcomponent_size]

                    mutant = a + self.mutation_factor * (b - c)
                    mutant = np.clip(mutant, *subcomponent_bounds)

                    cross_points = np.random.rand(self.subcomponent_size) < self.crossover_rate
                    if not np.any(cross_points):
                        cross_points[np.random.randint(0, self.subcomponent_size)] = True

                    trial_subcomponent = np.where(cross_points, mutant, population[i, start:start + self.subcomponent_size])
                    
                    # Enforce periodicity in subcomponents
                    period = np.random.randint(1, self.subcomponent_size // 2 + 1)
                    trial_subcomponent = np.tile(trial_subcomponent[:period], self.subcomponent_size // period + 1)[:self.subcomponent_size]

                    trial = np.copy(population[i])
                    trial[start:start + self.subcomponent_size] = trial_subcomponent

                    score = func(trial)
                    self.evaluations += 1

                    if score > func(population[i]):
                        population[i, start:start + self.subcomponent_size] = trial_subcomponent
                        if score > best_score:
                            best_solution, best_score = np.copy(trial), score
        
        return best_solution

    def gradient_based_refinement(self, func, best_solution, bounds):
        tol_factor = max(1, self.budget - self.evaluations)
        res = minimize(lambda x: -func(x), best_solution, bounds=bounds, method='L-BFGS-B', options={'ftol': 1e-6 / tol_factor})
        return res.x

    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        best_solution = self.cooperative_coevolution(func, bounds)
        best_solution = self.gradient_based_refinement(func, best_solution, bounds)
        return best_solution