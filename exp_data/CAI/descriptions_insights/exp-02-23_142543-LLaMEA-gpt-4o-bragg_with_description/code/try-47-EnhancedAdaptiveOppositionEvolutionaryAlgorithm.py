import numpy as np

class EnhancedAdaptiveOppositionEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.alpha = 0.5  # Opposition learning rate
        self.initial_mutation_rate = 0.1
        self.crossover_rate = 0.7
        self.local_search_probability = 0.3

    def local_search(self, ind, func, lb, ub):
        perturbation = np.random.uniform(-0.05, 0.05, self.dim) * (ub - lb)
        new_ind = np.clip(ind + perturbation, lb, ub)
        return new_ind if func(new_ind) < func(ind) else ind

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size
        mutation_rate = self.initial_mutation_rate

        while evaluations < self.budget:
            # Dynamic opposition-based learning
            opposite_population = lb + ub - population + np.random.normal(0, 0.2, population.shape)
            opposite_fitness = np.array([func(ind) for ind in opposite_population])
            evaluations += self.population_size

            # Adaptive parameter tuning with dynamic mutation
            self.crossover_rate = 0.5 + 0.5 * np.random.rand()
            mutation_rate = 0.05 + 0.1 * (self.budget - evaluations) / self.budget

            # Generate offspring
            offspring = []
            for i in range(self.population_size):
                if np.random.rand() < self.crossover_rate:
                    parents = np.random.choice(self.population_size, 2, replace=False)
                    parent1, parent2 = population[parents]
                    child = np.clip(parent1 + self.alpha * (parent2 - parent1), lb, ub)
                    if np.random.rand() < mutation_rate:
                        mutation = np.random.uniform(-1, 1, self.dim) * (ub - lb) * 0.05
                        child = np.clip(child + mutation, lb, ub)
                    offspring.append(child)
                
                # Local search phase
                if np.random.rand() < self.local_search_probability:
                    offspring[-1] = self.local_search(offspring[-1], func, lb, ub)

            offspring = np.array(offspring)
            offspring_fitness = np.array([func(ind) for ind in offspring])
            evaluations += len(offspring)

            # Combine and select next generation with elitism
            combined_population = np.vstack((population, opposite_population, offspring))
            combined_fitness = np.hstack((fitness, opposite_fitness, offspring_fitness))
            best_indices = np.argsort(combined_fitness)[:self.population_size]
            population = combined_population[best_indices]
            fitness = combined_fitness[best_indices]

        return population[np.argmin(fitness)]