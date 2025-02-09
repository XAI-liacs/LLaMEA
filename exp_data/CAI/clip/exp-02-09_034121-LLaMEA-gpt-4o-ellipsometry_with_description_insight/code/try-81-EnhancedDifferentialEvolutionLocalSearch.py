import numpy as np
from scipy.optimize import minimize

class EnhancedDifferentialEvolutionLocalSearch:
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

                # Crossover
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

            # Adaptive mutation and crossover adjustment
            fitness_std = fitness.std()
            fitness_mean = fitness.mean()
            if fitness_mean > 0:
                F = 0.5 * (1 - fitness_std / fitness_mean)
                CR = 0.9 * (1 - fitness_std / fitness_mean)

            # Hybrid local search using BFGS and Nelder-Mead
            if self.budget > 0:
                result_bfgs = minimize(func, best_solution, method='L-BFGS-B', bounds=bounds, options={'maxfun': self.budget})
                self.budget -= result_bfgs.nfev
                
                if result_bfgs.fun < best_value:
                    best_value = result_bfgs.fun
                    best_solution = result_bfgs.x

                if self.budget > 0:
                    result_nm = minimize(func, best_solution, method='Nelder-Mead', options={'maxfev': self.budget})
                    self.budget -= result_nm.nfev

                    if result_nm.fun < best_value:
                        best_value = result_nm.fun
                        best_solution = result_nm.x
        
        return best_solution