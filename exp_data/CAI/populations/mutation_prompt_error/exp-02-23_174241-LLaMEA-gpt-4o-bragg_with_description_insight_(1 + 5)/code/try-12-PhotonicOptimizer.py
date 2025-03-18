import numpy as np
from scipy.optimize import minimize

class PhotonicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim  # Heuristic for population size
        self.F = 0.5 + 0.3 * np.random.rand()  # Adaptive differential weight
        self.CR = 0.7 + 0.2 * np.random.rand()  # Adaptive crossover probability
        self.bounds = None
        
    def initialize_population(self):
        lb, ub = self.bounds.lb, self.bounds.ub
        return lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
    
    def mutate(self, population):
        idxs = np.random.choice(self.population_size, 3, replace=False)
        a, b, c = population[idxs]
        mutant = np.clip(a + self.F * (b - c), self.bounds.lb, self.bounds.ub)
        return mutant
    
    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial
    
    def periodicity_penalty(self, solution):
        period = self.dim // 2
        penalties = [(solution[i] - solution[i + period]) ** 2 for i in range(self.dim - period)]
        return np.sum(penalties)
    
    def __call__(self, func):
        self.bounds = func.bounds
        population = self.initialize_population()
        best_solution = None
        best_score = float('inf')
        
        eval_count = 0
        while eval_count < self.budget:
            new_population = np.zeros_like(population)
            for i in range(self.population_size):
                target = population[i]
                mutant = self.mutate(population)
                trial = self.crossover(target, mutant)
                
                trial_score = func(trial) + self.periodicity_penalty(trial)
                eval_count += 1
                
                if trial_score < best_score:
                    best_score = trial_score
                    best_solution = trial
                
                if trial_score < func(target) + self.periodicity_penalty(target):
                    new_population[i] = trial
                else:
                    new_population[i] = target
            
            population = new_population
            
            # Local optimization using BFGS if budget allows
            if eval_count < self.budget:
                result = minimize(func, best_solution, method='L-BFGS-B', bounds=zip(self.bounds.lb, self.bounds.ub))
                eval_count += result.nfev
                if result.fun < best_score:
                    best_score = result.fun
                    best_solution = result.x
        
        return best_solution