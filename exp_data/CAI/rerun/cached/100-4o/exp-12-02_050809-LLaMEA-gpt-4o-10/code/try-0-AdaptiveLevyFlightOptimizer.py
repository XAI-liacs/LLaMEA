import numpy as np

class AdaptiveLevyFlightOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.best_solution = None
        self.best_value = float('inf')
        self.population_size = 10 + int(0.1 * dim)
        self.step_size = 0.1  # Initial step size for mutations

    def levy_flight(self, size):
        # Generate Levy flight steps
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, size)
        v = np.random.normal(0, 1, size)
        step = u / np.abs(v) ** (1 / beta)
        return 0.01 * step

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        
        # Evaluate initial solutions
        for i in range(self.population_size):
            if fitness[i] < self.best_value:
                self.best_value = fitness[i]
                self.best_solution = population[i]

        evaluations = self.population_size
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                # Lévy flight
                step = self.levy_flight(self.dim)
                candidate = population[i] + self.step_size * step
                candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
                
                # Evaluate candidate
                candidate_value = func(candidate)
                evaluations += 1
                
                # Adaptive step size adjustment
                if candidate_value < fitness[i]:
                    fitness[i] = candidate_value
                    population[i] = candidate
                    if candidate_value < self.best_value:
                        self.best_value = candidate_value
                        self.best_solution = candidate
                else:
                    # Reduce step size to exploit local region
                    self.step_size *= 0.99

            # Gradually increase the step size to enhance exploration
            self.step_size = min(self.step_size * 1.01, 0.1)
        
        return self.best_solution, self.best_value