import numpy as np
from scipy.optimize import minimize

class HybridDEBFGS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = np.clip(5 * dim, 20, budget // 4)  # Adaptive population size
        self.mutation_factor = 0.8
        self.crossover_probability = 0.9

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        population[:self.population_size//2] = np.tile(np.mean([lb, ub], axis=0), (self.population_size//2, 1))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        while evaluations < self.budget:
            new_population = np.copy(population)
            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                adaptive_factor = max(self.mutation_factor * (1 - evaluations / self.budget), 0.5)
                mutant = population[a] + adaptive_factor * (population[b] - population[c])
                mutant = np.clip(mutant, lb, ub)
                adaptive_crossover = self.crossover_probability * (0.5 + 0.5 * np.tanh(0.1 * (self.budget - evaluations) / self.budget))
                cross_points = np.random.rand(self.dim) < adaptive_crossover 
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                trial_fitness = func(trial)
                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / (1 + np.abs(fitness[i]))):
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                evaluations += 1
                if evaluations >= self.budget:
                    break
            
            population = new_population

            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                periodic_length = np.random.randint(1, self.dim//2 + 1)  # Line changed
                periodic_solution = np.tile(population[i][:periodic_length], self.dim//periodic_length + 1)[:self.dim]  # Line changed
                periodic_solution = np.clip(periodic_solution, lb, ub)
                periodic_fitness = func(periodic_solution)
                if periodic_fitness < fitness[i]:
                    population[i] = periodic_solution
                    fitness[i] = periodic_fitness
                evaluations += 1

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        res = minimize(func, best_solution, bounds=zip(lb, ub), method='L-BFGS-B')
        return res.x