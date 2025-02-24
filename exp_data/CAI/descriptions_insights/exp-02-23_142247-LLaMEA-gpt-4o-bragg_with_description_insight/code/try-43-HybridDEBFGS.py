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
        avg_value = np.mean(individual) * 0.5 + np.median(individual) * 0.5
        individual[:] = avg_value
        return individual
    
    def differential_evolution(self, func, bounds):
        population = self.initialize_population(bounds)
        best_solution = None
        best_value = float('-inf')
        
        for gen in range(self.budget // self.pop_size):
            new_population = []
            dynamic_CR = self.CR * (1 - gen / (self.budget // self.pop_size)) * np.random.uniform(0.8, 1.2)  # Change 1: Slightly broaden range
            for i in range(self.pop_size):
                # Change 1: Randomly choose three different individuals to improve exploration
                a, b, c = population[np.random.choice(self.pop_size, 3, replace=False)]
                F = 0.5 + np.random.rand() * (1 - gen / (self.budget // self.pop_size))  # Adaptive F
                mutant = np.clip(a + F * (b - c), bounds.lb, bounds.ub)
                # Change 2: Modify crossover strategy to include adaptive probability
                trial = np.where(np.random.rand(self.dim) < dynamic_CR, mutant, population[i])
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
            population[np.random.choice(self.pop_size)] = best_solution
        return best_solution
    
    def refine_local(self, func, candidate, bounds):
        # Change 3: Introduce more flexible local bounds based optimization
        result = minimize(func, candidate, bounds=list(zip(bounds.lb, bounds.ub)), method='Powell')
        return result.x, result.fun
    
    def __call__(self, func):
        bounds = func.bounds
        best_candidate = self.differential_evolution(func, bounds)
        refined_solution, refined_value = self.refine_local(func, best_candidate, bounds)
        return refined_solution