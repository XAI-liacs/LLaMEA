import numpy as np
from scipy.optimize import minimize

class EnhancedDEBFGS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_factor = 0.8
        self.crossover_probability = 0.9

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        # Enhanced initialization with periodicity encouragement
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        for i in range(self.population_size):
            half_dim = self.dim // 2
            periodic_part = np.linspace(lb, ub, half_dim)
            population[i, :half_dim] = periodic_part
            population[i, half_dim:] = periodic_part

        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        while evaluations < self.budget:
            new_population = np.copy(population)
            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                # Adaptive mutation factor
                F = self.mutation_factor * (1 - evaluations / self.budget)
                mutant = population[a] + F * (population[b] - population[c])
                mutant = np.clip(mutant, lb, ub)
                cross_points = np.random.rand(self.dim) < self.crossover_probability
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                evaluations += 1
                if evaluations >= self.budget:
                    break
            
            population = new_population

            # Periodicity encouragement by modifying solutions
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                half_dim = self.dim // 2
                periodic_solution = np.tile(population[i][:half_dim], 2)
                periodic_solution = np.clip(periodic_solution, lb, ub)
                periodic_fitness = func(periodic_solution)
                if periodic_fitness < fitness[i]:
                    population[i] = periodic_solution
                    fitness[i] = periodic_fitness
                evaluations += 1

        # Refinement using local L-BFGS-B optimization
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        res = minimize(func, best_solution, bounds=[(lb[i], ub[i]) for i in range(self.dim)], method='L-BFGS-B')
        return res.x