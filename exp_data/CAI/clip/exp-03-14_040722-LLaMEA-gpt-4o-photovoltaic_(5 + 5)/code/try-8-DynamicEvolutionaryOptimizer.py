import numpy as np

class DynamicEvolutionaryOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.current_evaluations = 0

    def __call__(self, func):
        # Initialize population
        pop_size = 10 + 5 * self.dim
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.current_evaluations += pop_size

        # Evolutionary loop
        while self.current_evaluations < self.budget:
            # Select parents
            num_parents = pop_size // 2
            parents_indices = np.argsort(fitness)[:num_parents]
            parents = population[parents_indices]

            # Adaptive mutation strength based on fitness rank
            mutation_strength = ((func.bounds.ub - func.bounds.lb) / 10.0) * (1 - np.linspace(0, 0.5, num_parents)).reshape(-1, 1)
            
            # Generate offspring
            offspring = []
            for i in range(num_parents):
                parent = parents[i]
                mutated = parent + np.random.normal(0, mutation_strength[i], self.dim)
                mutated = np.clip(mutated, func.bounds.lb, func.bounds.ub)
                offspring.append(mutated)

            # Evaluate offspring
            offspring_fitness = np.array([func(ind) for ind in offspring])
            self.current_evaluations += len(offspring)

            # Elitism: Keep a proportion of the best from the previous generation
            elite_size = int(0.1 * pop_size)
            elite_indices = np.argsort(fitness)[:elite_size]
            elite = population[elite_indices]

            # Select the best individuals to form the new population
            population = np.vstack((elite, parents, offspring))
            fitness = np.hstack((fitness[elite_indices], fitness[parents_indices], offspring_fitness))
            best_indices = np.argsort(fitness)[:pop_size]
            population = population[best_indices]
            fitness = fitness[best_indices]

        # Return the best solution found
        best_index = np.argmin(fitness)
        return population[best_index]