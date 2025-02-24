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
        freq_factor = np.sin(np.linspace(0, np.pi, self.dim))  # Line change 1: Add frequency factor
        avg_value = np.mean(individual) * 0.5 + np.median(individual) * 0.5
        individual[:] = avg_value * freq_factor  # Line change 2: Multiply by frequency factor
        return individual
    
    def differential_evolution(self, func, bounds):
        population = self.initialize_population(bounds)
        best_solution = None
        best_value = float('-inf')
        
        for gen in range(self.budget // self.pop_size):
            new_population = []
            dynamic_CR = self.CR * (1 - gen / (self.budget // self.pop_size)) * np.random.uniform(0.9, 1.3)
            self.pop_size = max(4, int(self.pop_size * (0.9 ** gen)))
            for i in range(self.pop_size):
                a, b, c = population[np.random.choice(self.pop_size, 3, replace=False)]
                F = 0.5 + np.random.rand() * (0.5 + (1 - gen / (self.budget // self.pop_size))) * np.random.uniform(0.9, 1.1)
                mutant = np.clip(a + F * (b - c), bounds.lb, bounds.ub)
                trial = np.where(np.random.rand(self.dim) < dynamic_CR, mutant, population[i])
                trial = self.enforce_periodicity(trial)  # Line change 3: Consistent periodic enforcement
                trial_value = func(trial)
                new_population.append(trial if trial_value > func(population[i]) else population[i])  # Change 1: Adaptive trial selection
                if trial_value > best_value:
                    best_value = trial_value
                    best_solution = trial
            population = np.array(new_population)
            if np.random.rand() < 0.5 + 0.5 * (gen / (self.budget // self.pop_size)):  # Change 2: Dynamic elitism retention
                population[np.random.choice(self.pop_size)] = best_solution
        return best_solution
    
    def refine_local(self, func, candidate, bounds):
        result = minimize(func, candidate, bounds=list(zip(bounds.lb, bounds.ub)), method='BFGS', options={'gtol': 1e-6})
        return result.x, result.fun
    
    def __call__(self, func):
        bounds = func.bounds
        best_candidate = self.differential_evolution(func, bounds)
        refined_solution, refined_value = self.refine_local(func, best_candidate, bounds)
        return refined_solution