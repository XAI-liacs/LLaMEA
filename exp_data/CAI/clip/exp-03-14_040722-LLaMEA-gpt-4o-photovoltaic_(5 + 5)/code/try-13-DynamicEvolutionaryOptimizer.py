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

            # Generate offspring using crossover
            offspring = []
            for i in range(num_parents // 2):
                p1, p2 = parents[2 * i], parents[2 * i + 1]
                cross_point = np.random.randint(1, self.dim)
                child1 = np.concatenate((p1[:cross_point], p2[cross_point:]))
                child2 = np.concatenate((p2[:cross_point], p1[cross_point:]))
                offspring.extend([child1, child2])
                
                # Mutate offspring
                diversity = np.std(population, axis=0)  # Calculate diversity
                mutation_strength = (func.bounds.ub - func.bounds.lb) / 10.0 * diversity
                offspring[-2] += np.random.normal(0, mutation_strength, self.dim)
                offspring[-1] += np.random.normal(0, mutation_strength, self.dim)

            # Evaluate offspring
            offspring = [np.clip(child, func.bounds.lb, func.bounds.ub) for child in offspring]
            offspring_fitness = np.array([func(ind) for ind in offspring])
            self.current_evaluations += len(offspring)

            # Select the best individuals to form the new population
            population = np.vstack((parents, offspring))
            fitness = np.hstack((fitness[parents_indices], offspring_fitness))
            best_indices = np.argsort(fitness)[:pop_size]
            population = population[best_indices]
            fitness = fitness[best_indices]

        # Return the best solution found
        best_index = np.argmin(fitness)
        return population[best_index]