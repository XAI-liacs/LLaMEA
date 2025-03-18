import numpy as np

class HybridMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = 10 * self.dim  # Heuristic choice
        F = 0.8  # Mutation factor for differential evolution
        CR = 0.9  # Crossover probability
        np.random.seed(42)

        # Initialize population randomly within bounds
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size

        while evaluations < self.budget:
            for i in range(population_size):
                # Differential Evolution Mutation
                indices = list(range(population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                # Adjusted Mutation Strategy
                mutant = population[a] + F * (population[b] - population[c]) + 0.1 * (population[a] - population[i])
                mutant = np.clip(mutant, lb, ub)

                # Crossover
                crossover = np.random.rand(self.dim) < CR
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True
                trial = np.where(crossover, mutant, population[i])

                # Evaluate trial solution
                trial_fitness = func(trial)
                evaluations += 1

                # Select the better solution
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                # Adaptive Neighborhood Search
                if np.random.rand() < 0.1:  # 10% chance to perform local search
                    local_search_point = population[i]
                    perturbation = np.random.normal(0, 0.1, self.dim) * (ub - lb)
                    neighbor = local_search_point + perturbation
                    neighbor = np.clip(neighbor, lb, ub)
                    neighbor_fitness = func(neighbor)
                    evaluations += 1

                    if neighbor_fitness < fitness[i]:
                        population[i] = neighbor
                        fitness[i] = neighbor_fitness

                if evaluations >= self.budget:
                    break

        # Return the best found solution
        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]