import numpy as np

class EnhancedHybridMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = 10 * self.dim
        np.random.seed(42)

        # Initialize population and control parameters
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size

        F = 0.5 + 0.3 * np.random.rand(population_size)  # Self-adaptive mutation factor
        CR = 0.8 + 0.1 * np.random.rand(population_size)  # Self-adaptive crossover rate

        while evaluations < self.budget:
            for i in range(population_size):
                # Differential Evolution Mutation
                indices = list(range(population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = population[a] + F[i] * (population[b] - population[c])
                mutant = np.clip(mutant, lb, ub)

                # Crossover
                crossover = np.random.rand(self.dim) < CR[i]
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

                # Dynamic Local Search
                if np.random.rand() < 0.2:  # 20% chance to perform local search
                    intensity = 1.0 - (evaluations / self.budget)
                    perturbation = np.random.normal(0, 0.1 * intensity, self.dim) * (ub - lb)
                    neighbor = population[i] + perturbation
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