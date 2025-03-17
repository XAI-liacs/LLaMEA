import numpy as np

class DynamicEvolutionaryOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.current_evaluations = 0

    def __call__(self, func):
        # Initialize population with increased diversity
        pop_size = 12 + 6 * self.dim
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.current_evaluations += pop_size

        # Evolutionary loop
        while self.current_evaluations < self.budget:
            # Adjust population size based on convergence
            if self.current_evaluations % (self.budget // 5) == 0:
                pop_size += 2  # Increment population size periodically

            # Select parents
            num_parents = pop_size // 2
            parents_indices = np.argsort(fitness)[:num_parents]
            parents = population[parents_indices]

            # Adaptive mutation strength based on fitness variance
            fitness_variance = np.var(fitness)
            mutation_strength = ((func.bounds.ub - func.bounds.lb) / (12.0 + fitness_variance)) * \
                                (1 - np.linspace(0, 0.5, num_parents)).reshape(-1, 1) * np.random.uniform(0.8, 1.2, (num_parents, 1))
            
            # Generate offspring with adaptive crossover
            offspring = []
            for i in range(num_parents):
                parent1 = parents[i]
                parent2 = parents[np.random.choice(num_parents)]
                parent3 = parents[np.random.choice(num_parents)]  # Added third parent
                parent4 = parents[np.random.choice(num_parents)]  # Added fourth parent
                crossover_weights = np.random.dirichlet(np.ones(4), size=1).flatten()
                child = (crossover_weights[0] * parent1 + crossover_weights[1] * parent2 +
                         crossover_weights[2] * parent3 + crossover_weights[3] * parent4)
                mutated = child + np.random.normal(0, mutation_strength[i], self.dim)
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