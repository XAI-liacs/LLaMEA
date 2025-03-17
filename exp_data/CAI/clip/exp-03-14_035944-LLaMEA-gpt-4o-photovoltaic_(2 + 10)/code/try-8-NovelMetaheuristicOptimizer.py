import numpy as np

class NovelMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = 10
        mutate_scale = 0.1
        crossover_rate = 0.7
        neighborhood_size = 5

        # Initialize population
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size

        while evaluations < self.budget:
            # Select parents
            parents_idx = np.random.choice(range(population_size), size=neighborhood_size, replace=False)
            parents = population[parents_idx]
            best_parent_idx = np.argmin(fitness[parents_idx])
            best_parent = parents[best_parent_idx]
            
            # Crossover and Mutation
            for i in range(population_size):
                if np.random.rand() < crossover_rate:
                    other = parents[np.random.randint(0, neighborhood_size)]
                    child = best_parent + np.random.randn(self.dim) * mutate_scale * (best_parent - other)
                else:
                    child = best_parent + np.random.randn(self.dim) * mutate_scale
                
                child = np.clip(child, lb, ub)

                # Evaluate child
                child_fitness = func(child)
                evaluations += 1

                # Adjust mutation scale dynamically based on population diversity
                mutate_scale = 0.1 + 0.5 * np.std(population, axis=0).mean()

                # Replace worst individual if the child is better
                worst_idx = np.argmax(fitness)
                if child_fitness < fitness[worst_idx]:
                    population[worst_idx] = child
                    fitness[worst_idx] = child_fitness

                if evaluations >= self.budget:
                    break

        best_idx = np.argmin(fitness)
        return population[best_idx]