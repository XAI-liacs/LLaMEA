import numpy as np

class AdaptiveHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.temperature = 1.0
        self.num_parents = 5

    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        num_evaluations = 0

        # Initialize parent population randomly
        parents = np.random.uniform(bounds[0], bounds[1], (self.num_parents, self.dim))
        parent_fitness = np.array([func(ind) for ind in parents])
        num_evaluations += self.num_parents

        best_solution = parents[np.argmin(parent_fitness)]
        best_fitness = np.min(parent_fitness)

        while num_evaluations < self.budget:
            # Generate offspring through mutation
            offspring = []
            for parent in parents:
                if num_evaluations >= self.budget:
                    break
                candidate = parent + np.random.normal(0, self.temperature, self.dim)
                candidate = np.clip(candidate, bounds[0], bounds[1])
                offspring.append(candidate)
                num_evaluations += 1

            # Evaluate offspring
            offspring_fitness = np.array([func(ind) for ind in offspring])

            # Combine parents and offspring
            combined_population = np.vstack((parents, offspring))
            combined_fitness = np.hstack((parent_fitness, offspring_fitness))

            # Select the top num_parents individuals to form new parents
            selected_indices = np.argsort(combined_fitness)[:self.num_parents]
            parents = combined_population[selected_indices]
            parent_fitness = combined_fitness[selected_indices]

            # Update the best solution found
            if np.min(parent_fitness) < best_fitness:
                best_fitness = np.min(parent_fitness)
                best_solution = parents[np.argmin(parent_fitness)]

            # Simulated annealing: adapt temperature
            self.temperature *= 0.95  # Simple cooling strategy

        return best_solution