import numpy as np
from scipy.optimize import minimize

class EnhancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(30, budget // 10)
        self.strategy_switch = 0.2  # Switch to Differential Evolution after 20% of budget
        self.CMA_eta = 0.15  # Learning rate for covariance matrix adaptation

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
                    candidate = population[i] + np.random.normal(0, 0.1, self.dim)
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
                # Differential Evolution with adaptive mutation and rank-based selection
                population_fitness_pairs = list(zip(population, fitness))
                population_fitness_pairs.sort(key=lambda x: x[1])
                scale_factor = 0.5 + 0.1 * np.random.rand()
                for i in range(self.population_size):
                    candidates = np.random.choice(self.population_size, 3, replace=False, p=np.linspace(0.1, 0.9, self.population_size))
                    a, b, c = [population_fitness_pairs[idx][0] for idx in candidates]
                    mutant = a + scale_factor * (b - c)
                    mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                    trial = np.where(np.random.rand(self.dim) < 0.8, mutant, population[i])
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
                # Covariance Matrix Adaptation for mutation rate tuning
                mean_solution = np.mean(population, axis=0)
                deviation = np.std(population, axis=0)
                self.CMA_eta = np.clip(self.CMA_eta * (1.0 + 0.1 * np.random.randn()), 0.05, 0.3)
                population += self.CMA_eta * np.random.multivariate_normal(np.zeros(self.dim), np.diag(deviation), self.population_size)

        return best_solution