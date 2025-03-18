import numpy as np

class AdaptiveGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_rate = 0.1
        self.elitism_rate = 0.1
        self.global_best_position = None

    def __call__(self, func):
        lower_bound = func.bounds.lb
        upper_bound = func.bounds.ub
        
        # Initialize population
        population = np.random.uniform(lower_bound, upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size
        
        if self.global_best_position is None:
            best_index = np.argmin(fitness)
            self.global_best_position = population[best_index]
            global_best_fitness = fitness[best_index]

        while evaluations < self.budget:
            # Select parents using tournament selection
            parents = self.tournament_selection(population, fitness)

            # Generate offspring using crossover
            offspring = self.crossover(parents)

            # Apply mutation
            self.dynamic_mutation_rate(evaluations)
            offspring = self.mutation(offspring, lower_bound, upper_bound)

            # Evaluate fitness of offspring
            offspring_fitness = np.array([func(ind) for ind in offspring])
            evaluations += len(offspring)

            # Combine population and select next generation with elitism
            combined_population = np.vstack((population, offspring))
            combined_fitness = np.hstack((fitness, offspring_fitness))

            elite_count = int(self.elitism_rate * self.population_size)
            best_indices = np.argpartition(combined_fitness, elite_count)[:elite_count]
            population = combined_population[best_indices]
            fitness = combined_fitness[best_indices]

            # Update global best
            if fitness[0] < global_best_fitness:
                self.global_best_position = population[0]
                global_best_fitness = fitness[0]

            if evaluations >= self.budget:
                break

        return self.global_best_position, global_best_fitness

    def tournament_selection(self, population, fitness):
        # Tournament selection implementation
        selected_parents = []
        for _ in range(self.population_size):
            competitors_idx = np.random.choice(range(self.population_size), 3, replace=False)
            winner_idx = competitors_idx[np.argmin(fitness[competitors_idx])]
            selected_parents.append(population[winner_idx])
        return np.array(selected_parents)

    def crossover(self, parents):
        # Single-point crossover implementation
        offspring = []
        for i in range(0, self.population_size, 2):
            parent1, parent2 = parents[i], parents[i + 1]
            point = np.random.randint(1, self.dim)
            child1 = np.hstack((parent1[:point], parent2[point:]))
            child2 = np.hstack((parent2[:point], parent1[point:]))
            offspring.extend([child1, child2])
        return np.array(offspring)

    def mutation(self, offspring, lower_bound, upper_bound):
        # Mutation implementation with dynamic mutation rate
        for i in range(len(offspring)):
            if np.random.rand() < self.mutation_rate:
                mutate_dim = np.random.randint(0, self.dim)
                offspring[i][mutate_dim] += np.random.uniform(-1.0, 1.0) * (upper_bound[mutate_dim] - lower_bound[mutate_dim]) * 0.1
                offspring[i] = np.clip(offspring[i], lower_bound, upper_bound)
        return offspring

    def dynamic_mutation_rate(self, evaluations):
        # Adjust mutation rate based on progress
        self.mutation_rate = 0.1 * (1 - evaluations / self.budget)