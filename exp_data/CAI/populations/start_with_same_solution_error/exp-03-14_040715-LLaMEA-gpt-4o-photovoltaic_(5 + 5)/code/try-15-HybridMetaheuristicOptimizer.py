import numpy as np

class HybridMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        np.random.seed(42)  # For reproducibility
        bounds = func.bounds
        population_size = 10 * self.dim
        F = 0.5  # Differential weight
        CR = 0.9  # Crossover probability
        population = np.random.uniform(bounds.lb, bounds.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        budget_used = population_size

        while budget_used < self.budget:
            for i in range(population_size):
                # Mutation
                indices = list(range(population_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                F_dynamic = 0.5 + 0.3 * (1 - budget_used / self.budget)  # Dynamic F
                mutant = np.clip(a + F_dynamic * (b - c), bounds.lb, bounds.ub)
                
                # Crossover
                CR_dynamic = 0.9 - 0.5 * (budget_used / self.budget)  # Dynamic adjustment
                cross_points = np.random.rand(self.dim) < CR_dynamic
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                # Selection
                trial_fitness = func(trial)
                budget_used += 1
                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    population[i] = trial
                
                if budget_used >= self.budget:
                    break

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]