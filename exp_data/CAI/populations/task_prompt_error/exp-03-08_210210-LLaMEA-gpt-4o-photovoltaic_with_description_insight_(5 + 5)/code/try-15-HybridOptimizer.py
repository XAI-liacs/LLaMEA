import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.CR = 0.9
        self.F = 0.8
        self.evaluations = 0
    
    def differential_evolution(self, func, bounds):
        # Initialize population within bounds
        population = np.random.rand(self.pop_size, self.dim)
        population = bounds.lb + (bounds.ub - bounds.lb) * population
        fitness = np.array([func(ind) for ind in population])
        self.evaluations += self.pop_size
        
        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                if self.evaluations >= self.budget:
                    break
                
                # Mutation with adaptive F
                idxs = np.random.choice(self.pop_size, 3, replace=False)
                a, b, c = population[idxs]
                self.F = 0.5 + 0.3 * np.random.rand()  # Adjusted mutation factor
                mutant = np.clip(a + self.F * (b - c), bounds.lb, bounds.ub)
                
                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                # Selection
                trial_fitness = func(trial)
                self.evaluations += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
        
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]

    def local_refinement(self, func, x0, bounds):
        # Local optimization using L-BFGS-B
        result = minimize(func, x0, method='L-BFGS-B', bounds=[(l, u) for l, u in zip(bounds.lb, bounds.ub)])
        return result.x, result.fun

    def __call__(self, func):
        bounds = func.bounds
        best_solution, best_fitness = self.differential_evolution(func, bounds)
        if self.evaluations < self.budget:
            refined_solution, refined_fitness = self.local_refinement(func, best_solution, bounds)
            if refined_fitness < best_fitness:
                best_solution, best_fitness = refined_solution, refined_fitness
        return best_solution