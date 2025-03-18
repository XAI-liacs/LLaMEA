import numpy as np
from scipy.optimize import minimize

class MultiScaleRefinement:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = np.clip(5 * dim, 20, budget // 4)
        self.mutation_factor = 0.8
        self.crossover_probability = 0.9
        self.scale_factors = [0.5, 0.25, 0.1]  # Different scales for exploration

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        while evaluations < self.budget:
            new_population = np.copy(population)
            for scale in self.scale_factors:
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

                if evaluations >= self.budget:
                    break

                # Refining search with local exploration
                for i in range(self.population_size):
                    if evaluations >= self.budget:
                        break
                    local_search_center = new_population[i]
                    local_lb = np.clip(local_search_center - (ub - lb) * scale, lb, ub)
                    local_ub = np.clip(local_search_center + (ub - lb) * scale, lb, ub)
                    local_solution = np.random.uniform(local_lb, local_ub, self.dim)
                    local_fitness = func(local_solution)
                    if local_fitness < fitness[i]:
                        new_population[i] = local_solution
                        fitness[i] = local_fitness
                    evaluations += 1

            population = new_population

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        res = minimize(func, best_solution, bounds=zip(lb, ub), method='L-BFGS-B')
        return res.x