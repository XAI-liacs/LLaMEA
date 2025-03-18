import numpy as np
from scipy.optimize import minimize

class HybridSimulatedAnnealing:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.initial_temp = 100.0
        self.cooling_rate = 0.95
        self.mutation_scale = 0.1

    def initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        return lb + (ub - lb) * np.random.rand(self.population_size, self.dim)

    def simulated_annealing(self, x, func, bounds):
        current_temp = self.initial_temp
        current_solution = np.copy(x)
        current_obj = func(current_solution)
        
        while current_temp > 1 and self.evaluations < self.budget:
            # Adaptive mutation to promote exploration
            mutation = self.mutation_scale * np.random.randn(self.dim)
            new_solution = np.clip(current_solution + mutation, bounds.lb, bounds.ub)
            new_obj = func(new_solution)
            self.evaluations += 1
            
            # Acceptance probability
            if new_obj < current_obj or np.exp((current_obj - new_obj) / current_temp) > np.random.rand():
                current_solution = new_solution
                current_obj = new_obj
            
            # Cool down
            current_temp *= self.cooling_rate

        return current_solution

    def local_optimization(self, x0, func, bounds):
        result = minimize(func, x0, bounds=[(lb, ub) for lb, ub in zip(bounds.lb, bounds.ub)], method='L-BFGS-B')
        return result.x if result.success else x0

    def __call__(self, func):
        bounds = func.bounds
        population = self.initialize_population(bounds)
        self.evaluations = 0

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations < self.budget:
                    # Simulated Annealing
                    population[i] = self.simulated_annealing(population[i], func, bounds)
                    
                    if self.evaluations < self.budget:
                        # Local optimization
                        population[i] = self.local_optimization(population[i], func, bounds)
                        self.evaluations += 1

        # Return the best solution found
        best_idx = np.argmin([func(ind) for ind in population])
        return population[best_idx]