import numpy as np
from scipy.optimize import minimize

class HybridBraggOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = dim * 10
        self.mutation_factor = 0.8
        self.crossover_prob = 0.9
        self.local_search_prob = 0.25  # Adjusted local search probability
        self.current_evals = 0
        
    def __call__(self, func):
        bounds = func.bounds
        lb, ub = bounds.lb, bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        best_solution = None
        best_fitness = float('inf')

        def periodic_penalty(x):
            half_dim = self.dim // 2
            first_half = x[:half_dim]
            second_half = x[half_dim:]
            penalty = np.sum((first_half - second_half) ** 2)
            return penalty

        while self.current_evals < self.budget:
            if best_solution is None:
                fitness = np.apply_along_axis(func, 1, population)
                self.current_evals += self.population_size
                best_idx = np.argmin(fitness)
                best_fitness = fitness[best_idx]
                best_solution = population[best_idx]

            new_population = np.empty_like(population)
            for i in range(self.population_size):
                target = population[i]
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = population[indices]
                mutant = np.clip(a + self.mutation_factor * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < self.crossover_prob
                trial = np.where(cross_points, mutant, target)
                
                if self.current_evals < self.budget:
                    trial_fitness = func(trial) + periodic_penalty(trial)
                    self.current_evals += 1

                    if trial_fitness < fitness[i]:
                        new_population[i] = trial
                        fitness[i] = trial_fitness
                        if trial_fitness < best_fitness:
                            best_fitness = trial_fitness
                            best_solution = trial
                    else:
                        new_population[i] = target
                else:
                    break
                
            population = new_population

            if np.random.rand() < self.local_search_prob and self.current_evals < self.budget:
                result = minimize(func, best_solution, bounds=list(zip(lb, ub)), method='L-BFGS-B')
                if result.success and result.fun < best_fitness:
                    best_solution = result.x
                    best_fitness = result.fun
                self.current_evals += result.nfev

        return best_solution