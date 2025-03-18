import numpy as np
from scipy.optimize import minimize

class HybridDEBFGS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim  # Population size for DE
        self.F = 0.9  # Adjusted scaling factor for DE
        self.CR = 0.8  # Crossover probability for DE
    
    def initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        return lb + (ub - lb) * np.random.rand(self.pop_size, self.dim)
    
    def enforce_periodicity(self, individual):
        # Improved force periodicity by averaging all layers
        avg_value = np.mean(individual)
        individual[:] = avg_value
        return individual
    
    def differential_evolution(self, func, bounds):
        population = self.initialize_population(bounds)
        best_solution = None
        best_value = float('-inf')
        
        for _ in range(self.budget // self.pop_size):
            new_population = []
            for i in range(self.pop_size):
                a, b, c = population[np.random.choice(self.pop_size, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), bounds.lb, bounds.ub)
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, population[i])
                trial = self.enforce_periodicity(trial)
                trial_value = func(trial)
                if trial_value > func(population[i]):
                    new_population.append(trial)
                else:
                    new_population.append(population[i])
                if trial_value > best_value:
                    best_value = trial_value
                    best_solution = trial
            population = np.array(new_population)
        return best_solution
    
    def refine_local(self, func, candidate, bounds):
        result = minimize(func, candidate, bounds=list(zip(bounds.lb, bounds.ub)), method='L-BFGS-B')
        return result.x, result.fun
    
    def __call__(self, func):
        bounds = func.bounds
        # Step 1: Global search with Differential Evolution
        best_candidate = self.differential_evolution(func, bounds)
        # Step 2: Local refinement with BFGS
        refined_solution, refined_value = self.refine_local(func, best_candidate, bounds)
        return refined_solution