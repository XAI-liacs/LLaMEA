import numpy as np
from scipy.optimize import minimize

class RefinedHybridDEBFGS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_factor = 0.8
        self.crossover_probability = 0.9

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        # Symmetric initialization to promote periodicity
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        for i in range(self.population_size//2):
            base_pattern = np.random.uniform(lb, ub, self.dim//2)
            population[i] = np.tile(base_pattern, 2) if self.dim % 2 == 0 else np.hstack((np.tile(base_pattern, 2), base_pattern[:1]))
        
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        while evaluations < self.budget:
            new_population = np.copy(population)
            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = population[a] + self.mutation_factor * (population[b] - population[c])
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

            # Enhanced periodicity encouragement
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                base_pattern = population[i][:self.dim//2]
                periodic_solution = np.tile(base_pattern, 2) if self.dim % 2 == 0 else np.hstack((np.tile(base_pattern, 2), base_pattern[:1]))
                periodic_solution = np.clip(periodic_solution, lb, ub)
                periodic_fitness = func(periodic_solution)
                if periodic_fitness < fitness[i]:
                    population[i] = periodic_solution
                    fitness[i] = periodic_fitness
                evaluations += 1

        # Refinement with local BFGS optimization
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        res = minimize(func, best_solution, bounds=list(zip(lb, ub)), method='L-BFGS-B')
        return res.x