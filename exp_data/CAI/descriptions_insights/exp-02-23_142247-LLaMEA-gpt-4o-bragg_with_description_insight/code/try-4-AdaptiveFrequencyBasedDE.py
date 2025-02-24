import numpy as np
from scipy.optimize import minimize

class AdaptiveFrequencyBasedDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.initial_F = 0.5
        self.CR = 0.8
        self.history = []

    def initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        return lb + (ub - lb) * np.random.rand(self.pop_size, self.dim)
    
    def enforce_modularity(self, individual):
        # Ensure modular characteristics by segmenting into blocks and averaging
        block_size = self.dim // 4
        for i in range(0, self.dim, block_size):
            block_average = np.mean(individual[i:i + block_size])
            individual[i:i + block_size] = block_average
        return individual

    def adaptive_scaling(self, index):
        # Dynamic adjustment based on previous improvements
        if not self.history:
            return self.initial_F
        improved_counts = np.sum(self.history[-5:], axis=0)
        max_improved = np.max(improved_counts)
        return self.initial_F * (1 + (improved_counts[index] / max_improved) if max_improved > 0 else 0)

    def differential_evolution(self, func, bounds):
        population = self.initialize_population(bounds)
        best_solution = None
        best_value = float('-inf')
        
        for _ in range(self.budget // self.pop_size):
            new_population = []
            self.history.append([])
            for i in range(self.pop_size):
                a, b, c = population[np.random.choice(self.pop_size, 3, replace=False)]
                F = self.adaptive_scaling(i)
                mutant = np.clip(a + F * (b - c), bounds.lb, bounds.ub)
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, population[i])
                trial = self.enforce_modularity(trial)
                trial_value = func(trial)
                if trial_value > func(population[i]):
                    new_population.append(trial)
                    self.history[-1].append(1)  # Mark improvement
                else:
                    new_population.append(population[i])
                    self.history[-1].append(0)
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