import numpy as np
from scipy.optimize import minimize

class HybridMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.dim = dim
        self.bounds = None

    def quasi_oppositional_initialization(self, lb, ub):
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        quasi_opposite_population = lb + ub - population
        return np.concatenate((population, quasi_opposite_population), axis=0)

    def differential_evolution(self, func):
        np.random.seed(42)
        population = self.quasi_oppositional_initialization(self.bounds.lb, self.bounds.ub)
        population_fitness = np.array([func(ind) for ind in population])

        for _ in range(self.budget // (2 * self.population_size)):
            for i in range(self.population_size):
                indices = [idx for idx in range(len(population)) if idx != i]  # Fix line to use the correct length
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + 0.8 * (b - c), self.bounds.lb, self.bounds.ub)
                cross_points = np.random.rand(self.dim) < 0.7
                trial = np.where(cross_points, mutant, population[i])

                trial_fitness = func(trial)
                if trial_fitness < population_fitness[i]:
                    population[i] = trial
                    population_fitness[i] = trial_fitness

            best_idx = np.argmin(population_fitness)
            best_individual = population[best_idx]

            # Improved periodicity encouragement strategy
            if _ % 10 == 0:
                population = np.array([
                    0.5 * (population[i - 1] + population[i])
                    for i in range(1, len(population))  # Use correct length
                ])

        return best_individual

    def local_optimization(self, func, initial_guess):
        res = minimize(func, initial_guess, method='L-BFGS-B', bounds=list(zip(self.bounds.lb, self.bounds.ub)))
        return res.x if res.success else initial_guess

    def __call__(self, func):
        self.bounds = func.bounds
        best_global_solution = self.differential_evolution(func)
        best_solution = self.local_optimization(func, best_global_solution)
        return best_solution