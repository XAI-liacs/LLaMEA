import numpy as np
from scipy.optimize import minimize

class HybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.F = 0.8
        self.CR = 0.7
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.5
    
    def initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        return lb + (ub - lb) * np.random.rand(self.pop_size, self.dim)
    
    def enforce_periodicity(self, individual):
        period = 2
        for i in range(0, len(individual), period):
            individual[i:i+period] = np.mean(individual[i:i+period])
        return individual
    
    def differential_evolution(self, func, bounds, velocities):
        population = self.initialize_population(bounds)
        best_solution = None
        best_value = float('-inf')
        personal_best = np.copy(population)
        personal_best_values = np.array([func(ind) for ind in personal_best])
        
        for _ in range(self.budget // self.pop_size):
            new_population = []
            for i in range(self.pop_size):
                a, b, c = population[np.random.choice(self.pop_size, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), bounds.lb, bounds.ub)
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, population[i])
                trial = self.enforce_periodicity(trial)
                trial_value = func(trial)
                
                if trial_value > personal_best_values[i]:
                    personal_best[i] = trial
                    personal_best_values[i] = trial_value
                
                if trial_value > best_value:
                    best_value = trial_value
                    best_solution = trial
                
                velocities[i] = (self.w * velocities[i] + 
                                 self.c1 * np.random.rand(self.dim) * (personal_best[i] - population[i]) +
                                 self.c2 * np.random.rand(self.dim) * (best_solution - population[i]))
                
                population[i] = np.clip(population[i] + velocities[i], bounds.lb, bounds.ub)
                
            new_population = np.array(population)
            population = new_population
            
        return best_solution
    
    def refine_local(self, func, candidate, bounds):
        result = minimize(func, candidate, bounds=list(zip(bounds.lb, bounds.ub)), method='L-BFGS-B')
        return result.x, result.fun
    
    def __call__(self, func):
        bounds = func.bounds
        velocities = np.zeros((self.pop_size, self.dim))
        # Step 1: Global search with DE and PSO dynamics
        best_candidate = self.differential_evolution(func, bounds, velocities)
        # Step 2: Local refinement with BFGS
        refined_solution, refined_value = self.refine_local(func, best_candidate, bounds)
        return refined_solution