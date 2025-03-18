import numpy as np
from scipy.optimize import minimize

class AdaptivePeriodicDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10 * dim // 2, dim)
        self.scale_factor = 0.5
        self.crossover_probability = 0.9
        self.initial_periodicity_weight = 0.3
    
    def initialize_population(self, lb, ub):
        return np.random.uniform(lb, ub, (self.population_size, self.dim)) + 0.05 * (np.cos(np.arange(self.dim) * np.pi / self.dim))
    
    def adaptive_periodicity_cost(self, solution, gen):
        adaptive_weight = self.initial_periodicity_weight * (1 + 0.5 * np.sin(np.pi * gen / (self.budget // self.population_size)))
        return np.sum((solution - np.roll(solution, 1))**2) - adaptive_weight * np.cos(2 * np.pi * np.arange(self.dim) / self.dim).dot(solution)
    
    def differential_evolution_step(self, population, lb, ub, func, gen):
        new_population = np.empty_like(population)
        for i in range(self.population_size):
            idxs = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            scale_factor = self.scale_factor * (1 + 0.5 * np.cos(np.pi * gen / (self.budget // self.population_size)))
            mutant = np.clip(a + scale_factor * (b - c), lb, ub)
            cross_points = np.random.rand(self.dim) < (self.crossover_probability * (1 - gen / (self.budget // self.population_size)))
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, population[i])
            trial_with_periodicity = trial + self.adaptive_periodicity_cost(trial, gen)
            
            if func(trial_with_periodicity) < func(population[i]):
                new_population[i] = trial_with_periodicity
            else:
                new_population[i] = population[i]
        
        return new_population
    
    def multi_step_local_optimization(self, x0, func, lb, ub):
        result = x0
        for _ in range(3):  # Perform multiple local searches to refine solutions
            res = minimize(func, result, method='L-BFGS-B', tol=1e-5, bounds=[(lb[i], ub[i]) for i in range(self.dim)])
            if res.success and func(res.x) < func(result):
                result = res.x
        return result
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = self.initialize_population(lb, ub)
        best_solution = None
        best_score = float('inf')
        
        for gen in range(self.budget // self.population_size):
            population = self.differential_evolution_step(population, lb, ub, func, gen)
            for i in range(self.population_size):
                population[i] = self.multi_step_local_optimization(population[i], func, lb, ub)
                score = func(population[i])
                if score < best_score:
                    best_score = score
                    best_solution = population[i]
        
        return best_solution