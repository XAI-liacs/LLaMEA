import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = None
        self.ub = None
    
    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        population_size = 10 + 3 * self.dim
        F, CR = 0.5, 0.9
        population = self.initialize_population(population_size)
        fitness = np.array([func(ind) for ind in population])
        eval_count = population_size
        
        while eval_count < self.budget:
            for i in range(population_size):
                if eval_count >= self.budget:
                    break

                # DE Mutation with adaptive F
                indices = list(range(population_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                F = np.random.uniform(0.4, 0.9)  # Adaptively vary F
                mutant = np.clip(a + F * (b - c), self.lb, self.ub)

                # Enforce periodicity
                mutant = self.enforce_periodicity(mutant)

                # DE Crossover with adaptive CR
                CR = np.random.uniform(0.7, 1.0)  # Modified CR range
                trial = np.where(np.random.rand(self.dim) < CR, mutant, population[i])

                # Evaluate trial solution
                trial_fitness = func(trial)
                eval_count += 1

                # Select based on fitness
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

            # Local CMA-ES improvement
            if eval_count < self.budget:
                best_idx = np.argmin(fitness)
                best_solution = population[best_idx]
                result = minimize(func, best_solution, bounds=list(zip(self.lb, self.ub)), method='L-BFGS-B')
                eval_count += result.nfev
                if result.fun < fitness[best_idx]:
                    population[best_idx] = result.x
                    fitness[best_idx] = result.fun

        best_idx = np.argmin(fitness)
        return population[best_idx]

    def initialize_population(self, size):
        return np.random.uniform(self.lb, self.ub, (size, self.dim))

    def enforce_periodicity(self, vector):
        period = np.random.randint(2, 4)  # Allow flexible period lengths
        num_periods = self.dim // period
        for i in range(num_periods):
            mean_value = np.mean(vector[i*period:(i+1)*period])
            vector[i*period:(i+1)*period] = mean_value
        return vector