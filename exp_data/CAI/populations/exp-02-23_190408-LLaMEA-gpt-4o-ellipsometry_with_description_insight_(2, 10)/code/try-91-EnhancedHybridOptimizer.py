import numpy as np
from scipy.optimize import minimize

class EnhancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.exploration_factor = 0.1  # New line for dynamic boundary adjustment

    def __call__(self, func):
        bounds = list(zip(func.bounds.lb, func.bounds.ub))
        num_initial_samples = min(12 * self.dim, self.budget // 3)  # Changed line for better initial sampling
        samples = self.uniform_sampling(bounds, num_initial_samples)
        
        best_sample, best_value = None, float('inf')
        second_best_sample, second_best_value = None, float('inf')
        
        for sample in samples:
            value = func(sample)
            if value < best_value:
                second_best_sample, second_best_value = best_sample, best_value
                best_value, best_sample = value, sample
            elif value < second_best_value:
                second_best_value, second_best_sample = value, sample

        remaining_budget = self.budget - num_initial_samples
        allocated_budget_1 = int(remaining_budget * (0.6 + 0.1 * (best_value / max(second_best_value, 1e-9))))  # Changed line for better allocation

        res1 = self.local_optimization(func, best_sample, bounds, allocated_budget_1)
        
        # New block for swarm-based search
        swarm_search = self.swarm_search(func, second_best_sample, bounds, remaining_budget - allocated_budget_1)

        return (res1.x, res1.fun) if res1.fun < swarm_search[1] else swarm_search

    def uniform_sampling(self, bounds, num_samples):
        samples = []
        for _ in range(num_samples):
            sample = np.random.uniform([low - self.exploration_factor for low, _ in bounds],  # Changed factor
                                       [high + self.exploration_factor for _, high in bounds])  # Changed factor
            samples.append(np.clip(sample, [low for low, _ in bounds], [high for _, high in bounds]))
        return samples

    def local_optimization(self, func, initial_guess, bounds, budget):
        res = minimize(func, initial_guess, method='L-BFGS-B', bounds=bounds, options={'maxfun': budget, 'ftol': 1e-9})  # Changed tolerance
        return res

    def swarm_search(self, func, initial_sample, bounds, budget):  # New function for swarm-based backup search
        swarm_size = 5
        particles = [initial_sample + np.random.uniform(-self.exploration_factor, self.exploration_factor, self.dim) for _ in range(swarm_size)]
        best_particle, best_value = None, float('inf')
        for _ in range(budget // swarm_size):
            for particle in particles:
                value = func(np.clip(particle, *zip(*bounds)))  # Swarm particle evaluation
                if value < best_value:
                    best_value, best_particle = value, particle
        return best_particle, best_value