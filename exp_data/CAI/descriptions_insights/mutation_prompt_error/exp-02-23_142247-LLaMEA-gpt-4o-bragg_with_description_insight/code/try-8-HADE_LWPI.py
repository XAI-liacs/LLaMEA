import numpy as np
from scipy.optimize import minimize

class HADE_LWPI:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.F_min, self.F_max = 0.4, 0.9  # Adaptive scaling factor range
        self.CR_min, self.CR_max = 0.7, 0.9  # Adaptive crossover range
    
    def initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        return lb + (ub - lb) * np.random.rand(self.pop_size, self.dim)
    
    def adapt_parameters(self, gen):
        # Adapt parameters based on generation number
        F = self.F_min + (self.F_max - self.F_min) * (1 - np.exp(-gen / 100))
        CR = self.CR_max - (self.CR_max - self.CR_min) * (1 - np.exp(-gen / 50))
        return F, CR
    
    def inject_periodicity(self, individual):
        # Enforce periodicity by averaging pairs of adjacent layers
        for i in range(0, self.dim, 2):
            avg_value = (individual[i] + individual[(i + 1) % self.dim]) / 2
            individual[i] = avg_value
            individual[(i + 1) % self.dim] = avg_value
        return individual
    
    def differential_evolution(self, func, bounds):
        population = self.initialize_population(bounds)
        best_solution = None
        best_value = float('-inf')
        
        for gen in range(self.budget // self.pop_size):
            F, CR = self.adapt_parameters(gen)
            new_population = []
            for i in range(self.pop_size):
                a, b, c = population[np.random.choice(self.pop_size, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), bounds.lb, bounds.ub)
                trial = np.where(np.random.rand(self.dim) < CR, mutant, population[i])
                trial = self.inject_periodicity(trial)
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
        best_candidate = self.differential_evolution(func, bounds)
        refined_solution, refined_value = self.refine_local(func, best_candidate, bounds)
        return refined_solution