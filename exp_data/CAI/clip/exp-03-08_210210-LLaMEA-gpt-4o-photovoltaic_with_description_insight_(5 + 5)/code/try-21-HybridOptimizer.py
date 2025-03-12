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
        self.layer_increase_steps = [10, 20, 32]  # Change: Adaptive layer increase
    
    def differential_evolution(self, func, bounds):
        population = np.random.rand(self.pop_size, self.dim)
        population = bounds.lb + (bounds.ub - bounds.lb) * population
        fitness = np.array([func(ind) for ind in population])
        self.evaluations += self.pop_size
        
        while self.evaluations < self.budget:
            self.F = 0.5 + 0.3 * (1 - 0.7 * self.evaluations / self.budget)  # Change: Adjust F more aggressively
            for i in range(self.pop_size):
                if self.evaluations >= self.budget:
                    break
                
                idxs = np.random.choice(self.pop_size, 3, replace=False)
                a, b, c = population[idxs]
                modularity_weights = self.detect_layer_modularity(b, c)  # Change: Modularity detection
                mutant = np.clip(a + self.F * (b - c) * modularity_weights, bounds.lb, bounds.ub)
                
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                trial_fitness = func(trial)
                self.evaluations += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
            
            if self.evaluations < self.budget:
                self.increase_layers(func)  # Change: Dynamic layer increase

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]

    def detect_layer_modularity(self, b, c):
        # Simplified modularity detection
        modularity = np.where(np.abs(b - c) < 0.1, 0.5, 1.0)
        return modularity

    def increase_layers(self, func):
        # Change: Simplified layer increase logic
        for step in self.layer_increase_steps:
            if self.evaluations < self.budget:
                self.dim = step
                break

    def local_refinement(self, func, x0, bounds):
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