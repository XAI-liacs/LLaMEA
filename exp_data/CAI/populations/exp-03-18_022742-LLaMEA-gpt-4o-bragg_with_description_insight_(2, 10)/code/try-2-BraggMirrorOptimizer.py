import numpy as np
from scipy.optimize import minimize

class BraggMirrorOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.mutation_factor = 0.8
        self.crossover_probability = 0.7
        self.evaluations = 0

    def generate_initial_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        return np.random.uniform(lb, ub, (self.population_size, self.dim))
    
    def differential_evolution(self, population, bounds, func):
        for idx in range(self.population_size):
            # Choose three random individuals different from idx
            indices = [i for i in range(self.population_size) if i != idx]
            a, b, c = population[np.random.choice(indices, 3, replace=False)]
            
            # Differential mutation
            mutant = np.clip(a + self.mutation_factor * (b - c), bounds.lb, bounds.ub)
            
            # Crossover
            cross_points = np.random.rand(self.dim) < self.crossover_probability
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            
            trial = np.where(cross_points, mutant, population[idx])
            
            # Periodicity constraint promotion
            trial = self.promote_periodicity(trial, bounds)
            
            # Selection
            if self.evaluate(func, trial) < self.evaluate(func, population[idx]):
                population[idx] = trial
        
        return population
    
    def promote_periodicity(self, solution, bounds):
        period = 2  # Encourage periodicity with a fixed period size
        for i in range(len(solution) - period):
            solution[i] = 0.6 * solution[i] + 0.4 * solution[i + period]
        return np.clip(solution, bounds.lb, bounds.ub)
    
    def evaluate(self, func, solution):
        if self.evaluations < self.budget:
            self.evaluations += 1
            return func(solution)
        else:
            return np.inf
    
    def local_search(self, solution, func, bounds):
        result = minimize(func, solution, bounds=[(l, u) for l, u in zip(bounds.lb, bounds.ub)], method='L-BFGS-B')
        return result.x if result.success else solution
    
    def __call__(self, func):
        bounds = func.bounds
        population = self.generate_initial_population(bounds)
        
        best_solution = None
        best_score = np.inf
        
        while self.evaluations < self.budget:
            population = self.differential_evolution(population, bounds, func)
            for individual in population:
                improved_solution = self.local_search(individual, func, bounds)
                score = self.evaluate(func, improved_solution)
                if score < best_score:
                    best_score = score
                    best_solution = improved_solution
        
        return best_solution