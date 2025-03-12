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
        population = np.random.rand(self.pop_size, self.dim)
        population = bounds.lb + (bounds.ub - bounds.lb) * population
        fitness = np.array([self.evaluate_individual(func, ind) for ind in population])
        self.evaluations += self.pop_size
        
        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                if self.evaluations >= self.budget:
                    break
                
                idxs = np.random.choice(self.pop_size, 3, replace=False)
                a, b, c = population[idxs]
                mutant = np.clip(a + self.F * (b - c), bounds.lb, bounds.ub)
                
                cross_points = np.random.rand(self.dim) < self.variable_crossover_rate(i, fitness)
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                trial_fitness = self.evaluate_individual(func, trial)
                self.evaluations += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
        
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]

    def local_refinement(self, func, x0, bounds):
        result = minimize(func, x0, method='L-BFGS-B', bounds=[(l, u) for l, u in zip(bounds.lb, bounds.ub)])
        return result.x, result.fun

    def evaluate_individual(self, func, ind):
        perturbation = 0.001 * (np.random.rand(self.dim) - 0.5)
        return func(ind) + func(ind + perturbation)
    
    def variable_crossover_rate(self, index, fitness):
        return self.CR * (1 - (fitness[index] / (np.max(fitness) + 1e-8)))

    def __call__(self, func):
        bounds = func.bounds
        best_solution, best_fitness = self.differential_evolution(func, bounds)
        if self.evaluations < self.budget:
            refined_solution, refined_fitness = self.local_refinement(func, best_solution, bounds)
            if refined_fitness < best_fitness:
                best_solution, best_fitness = refined_solution, refined_fitness
        return best_solution