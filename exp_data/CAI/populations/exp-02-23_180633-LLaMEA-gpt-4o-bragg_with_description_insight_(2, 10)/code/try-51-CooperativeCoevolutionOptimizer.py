import numpy as np
from scipy.optimize import minimize

class CooperativeCoevolutionOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.subcomponent_size = self.dim // 2
        self.population_size = 20
        self.bounds = None

    def initialize_population(self, lb, ub):
        return np.random.uniform(lb, ub, (self.population_size, self.subcomponent_size))

    def evaluate_full_solution(self, func, subcomponent_a, subcomponent_b):
        full_solution = np.concatenate((subcomponent_a, subcomponent_b))
        return func(full_solution)

    def evolutionary_strategy(self, func, subcomponent):
        np.random.seed(42)
        population = self.initialize_population(self.bounds.lb[:self.subcomponent_size], self.bounds.ub[:self.subcomponent_size])
        population_fitness = np.array([self.evaluate_full_solution(func, ind, ind) for ind in population])

        for generation in range(self.budget // (2 * self.population_size)):
            F = 0.5 + (0.9 - 0.5) * generation / (self.budget // (2 * self.population_size))
            CR = 0.5 + (0.9 - 0.5) * generation / (self.budget // (2 * self.population_size))
            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), self.bounds.lb[:self.subcomponent_size], self.bounds.ub[:self.subcomponent_size])
                cross_points = np.random.rand(self.subcomponent_size) < CR
                trial = np.where(cross_points, mutant, population[i])

                trial_fitness = self.evaluate_full_solution(func, trial, trial)
                if trial_fitness < population_fitness[i]:
                    population[i] = trial
                    population_fitness[i] = trial_fitness

        best_idx = np.argmin(population_fitness)
        return population[best_idx]

    def local_optimization(self, func, initial_guess):
        res = minimize(func, initial_guess, method='L-BFGS-B', bounds=list(zip(self.bounds.lb, self.bounds.ub)))
        return res.x if res.success else initial_guess

    def __call__(self, func):
        self.bounds = func.bounds
        subcomponent_a = self.evolutionary_strategy(func, np.zeros(self.subcomponent_size))
        subcomponent_b = self.evolutionary_strategy(func, subcomponent_a)
        
        best_coevolved_solution = np.concatenate((subcomponent_a, subcomponent_b))
        best_solution = self.local_optimization(func, best_coevolved_solution)
        return best_solution