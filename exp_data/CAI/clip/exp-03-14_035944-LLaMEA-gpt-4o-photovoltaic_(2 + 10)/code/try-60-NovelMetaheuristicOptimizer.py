import numpy as np

class NovelMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        # Dynamic population size adaptation
        initial_population_size = 10
        max_population_size = 20
        min_population_size = 5
        population_size = initial_population_size
        mutate_scale = 0.1
        crossover_rate = 0.7
        neighborhood_size = 5

        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size

        while evaluations < self.budget:
            parents_idx = np.random.choice(range(population_size), size=neighborhood_size, replace=False)
            parents = population[parents_idx]
            best_parent_idx = np.argmin(fitness[parents_idx])
            best_parent = parents[best_parent_idx]
            
            # Introduce adaptive exploration-exploitation balance
            mutate_scale = 0.1 + 0.5 * np.std(fitness) / (np.mean(fitness) + 1e-9)

            for i in range(population_size):
                if np.random.rand() < crossover_rate:
                    other = parents[np.random.randint(0, neighborhood_size)]
                    child = best_parent + np.random.randn(self.dim) * mutate_scale * (best_parent - other)
                else:
                    child = best_parent + np.random.randn(self.dim) * mutate_scale
                
                child = np.clip(child, lb, ub)
                child_fitness = func(child)
                evaluations += 1

                # Update population size dynamically
                if evaluations % 10 == 0:
                    if np.std(fitness) < 0.01:
                        population_size = max(min_population_size, population_size - 1)
                    else:
                        population_size = min(max_population_size, population_size + 1)

                worst_idx = np.argmax(fitness)
                if child_fitness < fitness[worst_idx]:
                    population[worst_idx] = child
                    fitness[worst_idx] = child_fitness

                if evaluations >= self.budget:
                    break

        best_idx = np.argmin(fitness)
        return population[best_idx]