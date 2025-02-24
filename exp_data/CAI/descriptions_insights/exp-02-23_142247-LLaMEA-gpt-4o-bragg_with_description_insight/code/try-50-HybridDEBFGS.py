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
        pop = lb + (ub - lb) * np.random.rand(self.pop_size, self.dim)
        opp_pop = ub - (pop - lb)  # Quasi-oppositional initialization
        return np.vstack((pop, opp_pop))[:self.pop_size]

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
            dynamic_CR = self.CR * (1 - gen / (self.budget // self.pop_size)) * np.random.uniform(0.8, 1.2)
            for i in range(self.pop_size):
                a, b, c = population[np.random.choice(self.pop_size, 3, replace=False)]
                adaptive_F = self.F * np.random.uniform(0.5, 1.0)  # Adaptive scaling factor
                mutant = np.clip(a + adaptive_F * (b - c), bounds.lb, bounds.ub)
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
            population[np.random.choice(self.pop_size)] = best_solution  # Added: Elitism to retain best
        return best_solution
    
    def refine_local(self, func, candidate, bounds):
        result = minimize(func, candidate, bounds=list(zip(bounds.lb, bounds.ub)), method='BFGS')  # Changed to BFGS
        return result.x, result.fun
    
    def __call__(self, func):
        bounds = func.bounds
        best_candidate = self.differential_evolution(func, bounds)
        refined_solution, refined_value = self.refine_local(func, best_candidate, bounds)
        return refined_solution