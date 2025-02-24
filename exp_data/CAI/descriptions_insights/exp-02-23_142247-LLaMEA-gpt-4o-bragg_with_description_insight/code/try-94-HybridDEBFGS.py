import numpy as np
from scipy.optimize import minimize

class HybridDEBFGS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.F = 0.9
        self.CR = 0.8
    
    def initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        return lb + (ub - lb) * np.random.rand(self.pop_size, self.dim)
    
    def differential_evolution(self, func, bounds):
        population = self.initialize_population(bounds)
        best_solution = None
        best_value = float('-inf')
        
        for gen in range(self.budget // self.pop_size):
            new_population = []
            dynamic_CR = self.CR * (0.5 + 0.5 * np.cos(gen / (self.budget // self.pop_size) * np.pi))  # Change 1
            self.pop_size = max(4, int(self.pop_size * (0.95 ** gen)))  # Change 2
            for i in range(self.pop_size):
                a, b, c = population[np.random.choice(self.pop_size, 3, replace=False)]
                F = 0.5 + np.random.rand() * (0.6 + 0.4 * np.cos(gen / (self.budget // self.pop_size) * np.pi))  # Change 3
                mutant = np.clip(a + F * (b - c), bounds.lb, bounds.ub)
                trial = np.where(np.random.rand(self.dim) < dynamic_CR, mutant, population[i])
                if np.random.rand() < 0.5:  # Change 4
                    trial = self.enforce_periodicity(trial)
                trial_value = func(trial)
                new_population.append(trial if trial_value > func(population[i]) else population[i])
                if trial_value > best_value:
                    best_value = trial_value
                    best_solution = trial
            population = np.array(new_population)
            if gen % 3 == 0:  # Change 5
                population[np.random.choice(self.pop_size)] = best_solution
        return best_solution
    
    def enforce_periodicity(self, individual):
        avg_value = np.mean(individual) * 0.5 + np.median(individual) * 0.5
        individual[:] = avg_value
        return individual
    
    def refine_local(self, func, candidate, bounds):
        result = minimize(func, candidate, bounds=list(zip(bounds.lb, bounds.ub)), method='BFGS', options={'gtol': 1e-6})
        return result.x, result.fun
    
    def __call__(self, func):
        bounds = func.bounds
        best_candidate = self.differential_evolution(func, bounds)
        refined_solution, refined_value = self.refine_local(func, best_candidate, bounds)
        return refined_solution