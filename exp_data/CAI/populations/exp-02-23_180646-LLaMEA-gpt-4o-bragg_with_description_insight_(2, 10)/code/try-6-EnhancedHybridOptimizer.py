import numpy as np
from scipy.optimize import minimize

class EnhancedHybridOptimizer:
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

                # DE Mutation
                indices = list(range(population_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), self.lb, self.ub)

                # Enforce adaptive periodicity
                mutant = self.adaptive_periodicity(mutant, eval_count)

                # DE Crossover
                trial = np.where(np.random.rand(self.dim) < CR, mutant, population[i])

                # Evaluate trial solution
                trial_fitness = func(trial)
                eval_count += 1

                # Select based on fitness
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

            # Adaptive local search improvement
            if eval_count < self.budget:
                best_idx = np.argmin(fitness)
                best_solution = population[best_idx]
                result = minimize(func, best_solution, bounds=list(zip(self.lb, self.ub)), method='L-BFGS-B', options={'maxiter': self.adaptive_step(eval_count)})
                eval_count += result.nfev
                if result.fun < fitness[best_idx]:
                    population[best_idx] = result.x
                    fitness[best_idx] = result.fun

        best_idx = np.argmin(fitness)
        return population[best_idx]

    def initialize_population(self, size):
        return np.random.uniform(self.lb, self.ub, (size, self.dim))

    def adaptive_periodicity(self, vector, eval_count):
        # Adjust periodicity based on evaluation count
        period = 2 + (eval_count // (self.budget // 10)) % self.dim
        num_periods = self.dim // period
        for i in range(num_periods):
            mean_value = np.mean(vector[i*period:(i+1)*period])
            vector[i*period:(i+1)*period] = mean_value
        return vector
    
    def adaptive_step(self, eval_count):
        # Adjust the local search step size based on evaluation progress
        return max(1, self.budget // (eval_count // 10 + 1))