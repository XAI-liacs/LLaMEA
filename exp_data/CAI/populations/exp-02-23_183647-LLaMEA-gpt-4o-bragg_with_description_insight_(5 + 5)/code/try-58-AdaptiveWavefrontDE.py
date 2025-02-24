import numpy as np
from scipy.optimize import minimize

class AdaptiveWavefrontDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10 * dim // 2, dim)  # Adaptive population size
        self.scale_factor = 0.5
        self.crossover_probability = 0.9
        self.index_adjustment_weight = 0.1
    
    def initialize_population(self, lb, ub):
        return np.random.uniform(lb, ub, (self.population_size, self.dim))
    
    def refractive_index_adjustment(self, solution, gen):
        phase_shift = (np.cos(2 * np.pi * np.arange(self.dim) / self.dim) +
                       np.sin(np.pi * gen / (self.budget // self.population_size)))
        return solution + self.index_adjustment_weight * phase_shift
    
    def differential_evolution_step(self, population, lb, ub, func, gen):
        new_population = np.empty_like(population)
        self.crossover_probability = 0.9 - 0.5 * (gen / (self.budget // self.population_size))  # Line changed
        for i in range(self.population_size):
            idxs = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            scale_factor = self.scale_factor * np.cos(np.pi * gen / (self.budget // self.population_size))
            mutant = np.clip(a + scale_factor * (b - c), lb, ub)
            cross_points = np.random.rand(self.dim) < self.crossover_probability
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, population[i])
            trial_with_adjustment = self.refractive_index_adjustment(trial, gen)
            
            if func(trial_with_adjustment) < func(population[i]):
                new_population[i] = trial_with_adjustment
            else:
                new_population[i] = population[i]
        
        return new_population
    
    def local_optimization(self, x0, func, lb, ub):
        res = minimize(func, x0, method='L-BFGS-B', bounds=[(lb[i], ub[i]) for i in range(self.dim)])
        return res.x if res.success else x0
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = self.initialize_population(lb, ub)
        best_solution = None
        best_score = float('inf')
        
        for gen in range(self.budget // self.population_size):
            population = self.differential_evolution_step(population, lb, ub, func, gen)
            for i in range(self.population_size):
                population[i] = self.local_optimization(population[i], func, lb, ub)
                score = func(population[i])
                if score < best_score:
                    best_score = score
                    best_solution = population[i]
        
        return best_solution