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
            indices = [i for i in range(self.population_size) if i != idx]
            a, b, c = population[np.random.choice(indices, 3, replace=False)]
            
            # Changed line: enhanced chaotic mutation factor adaptation
            chaotic_mutation_factor = self.mutation_factor * (0.5 + 0.5 * np.sin(4 * np.pi * self.evaluations / self.budget))
            mutant = np.clip(a + chaotic_mutation_factor * ((b + c) / 2 - a), bounds.lb, bounds.ub)
            
            cross_points = np.random.rand(self.dim) < self.adaptive_crossover(idx)
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            
            trial = np.where(cross_points, mutant, population[idx])
            
            trial = self.promote_periodicity(trial, bounds)
            
            if self.evaluate(func, trial) < self.evaluate(func, population[idx]):
                population[idx] = trial
        
        return population
    
    def promote_periodicity(self, solution, bounds):
        # Changed line: refined modular preservation strategy
        period = max(2, int(self.dim * np.sin(2 * np.pi * (self.evaluations + 1) / self.budget)))
        for i in range(len(solution) - period):
            solution[i] = (solution[i] + solution[i + period]) / 2
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

    def adaptive_crossover(self, idx):
        # Changed line: refined chaotic crossover adaptation
        chaotic_value = 0.7 + 0.3 * np.sin(4 * np.pi * idx / self.population_size + self.evaluations / self.budget)
        return self.crossover_probability * chaotic_value

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