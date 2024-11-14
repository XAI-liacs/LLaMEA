import numpy as np
from scipy.optimize import minimize

class EnhancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(30, budget // 10)
        self.strategy_switch = 0.25  # Switch to Differential Evolution after 25% of budget

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )
        fitness = np.apply_along_axis(func, 1, population)
        evals = self.population_size
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        while evals < self.budget:
            if evals < self.strategy_switch * self.budget:
                # Random Search with Nelder-Mead and elitism
                for i in range(self.population_size):
                    candidate = population[i] + np.random.normal(0, 0.2, self.dim)
                    candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
                    candidate_fitness = func(candidate)
                    evals += 1
                    if candidate_fitness < fitness[i]:
                        population[i] = candidate
                        fitness[i] = candidate_fitness
                        if candidate_fitness < best_fitness:
                            best_fitness = candidate_fitness
                            best_solution = candidate.copy()
                    if evals >= self.budget:
                        break
                if evals + self.dim + 1 <= self.budget:
                    result = minimize(func, best_solution, method='Nelder-Mead', options={'maxfev': self.dim + 1})
                    evals += result.nfev
                    if result.fun < best_fitness:
                        best_fitness = result.fun
                        best_solution = result.x
            else:
                # Differential Evolution with adaptive mutation and modified crossover strategy
                scale_factor = 0.5 + 0.3 * np.random.rand()
                crossover_rate = 0.7 + 0.2 * np.random.rand()  # Modified crossover rate
                for i in range(self.population_size):
                    idxs = np.random.choice(self.population_size, 3, replace=False)
                    a, b, c = population[idxs]
                    mutant = a + scale_factor * (b - c)
                    mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                    crossover_vector = np.random.rand(self.dim) < crossover_rate
                    trial = np.where(crossover_vector, mutant, population[i])
                    trial_fitness = func(trial)
                    evals += 1
                    if trial_fitness < fitness[i]:
                        population[i] = trial
                        fitness[i] = trial_fitness
                        if trial_fitness < best_fitness:
                            best_fitness = trial_fitness
                            best_solution = trial.copy()
                    if evals >= self.budget:
                        break
                # Dynamic population adjustment
                if evals % (self.budget // 10) == 0:  # Adjust population size every 10% of budget
                    improvement = np.std(fitness) / np.abs(np.mean(fitness))
                    if improvement < 0.01 and self.population_size < self.budget // 5:
                        self.population_size = min(self.population_size + 5, int(self.budget / 10))

        return best_solution