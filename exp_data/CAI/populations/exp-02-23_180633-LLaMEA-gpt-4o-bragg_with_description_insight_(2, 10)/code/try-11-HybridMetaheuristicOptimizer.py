import numpy as np
from scipy.optimize import minimize

class HybridMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, dim)  # Adaptive population size
        self.bounds = None
        self.mutation_factor = 0.9
        self.crossover_rate = 0.7

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
                indices = [idx for idx in range(2 * self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), self.bounds.lb, self.bounds.ub)
                cross_points = np.random.rand(self.dim) < self.crossover_rate
                trial = np.where(cross_points, mutant, population[i])

                trial_fitness = func(trial)
                if trial_fitness < population_fitness[i]:
                    population[i] = trial
                    population_fitness[i] = trial_fitness

            best_idx = np.argmin(population_fitness)
            best_individual = population[best_idx]

            if _ % 10 == 0:  # Encourage periodicity more frequently
                population = np.array([
                    np.mean([population[i - 2], population[i - 1], population[i]], axis=0)
                    for i in range(2, self.population_size)
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