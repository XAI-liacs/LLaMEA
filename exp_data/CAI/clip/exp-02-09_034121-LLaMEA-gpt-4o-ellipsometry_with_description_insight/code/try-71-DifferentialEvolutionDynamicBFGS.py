import numpy as np
from scipy.optimize import minimize

class DifferentialEvolutionDynamicBFGS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        pop_size = max(10, self.dim * 5)
        bounds = np.array([(func.bounds.lb[i], func.bounds.ub[i]) for i in range(self.dim)])
        
        # Initialize population
        population = np.random.uniform(bounds[:, 0], bounds[:, 1], (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.budget -= pop_size
        
        best_idx = np.argmin(fitness)
        best_value = fitness[best_idx]
        best_solution = population[best_idx]
        
        F = 0.5  # Initial mutation factor
        CR = 0.9  # Initial crossover rate

        while self.budget > 0:
            for i in range(pop_size):
                if self.budget <= 0:
                    break
                
                # Mutation with dynamic adjustment
                indices = np.random.choice(pop_size, 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant = np.clip(x1 + F * (x2 - x3), bounds[:, 0], bounds[:, 1])

                # Crossover with dynamic adjustment
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                trial_fitness = func(trial)
                self.budget -= 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < best_value:
                        best_value = trial_fitness
                        best_solution = trial

            # Dynamic mutation and crossover adjustment based on fitness variance
            fitness_std = fitness.std()
            fitness_mean = fitness.mean()
            F = max(0.1, 0.5 * (1 - fitness_std / fitness_mean))
            CR = min(0.9, 0.5 + fitness_std / (fitness_mean + 1e-5))

            # Adaptive BFGS local search on the best solution found so far
            if self.budget > 0:
                result = minimize(func, best_solution, method='L-BFGS-B', bounds=bounds, options={'maxfun': min(self.budget, 100)})
                if result.fun < best_value:
                    best_value = result.fun
                    best_solution = result.x
                    self.budget -= result.nfev
        
        return best_solution