import numpy as np

class AdaptiveNeuroEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, dim)
        self.learning_rate = 0.1  # Initial learning rate for adaptive updates
        self.mutation_strength = 0.1  # Initial mutation strength
        self.crossover_rate = 0.7  # Probability of crossover
        self.adaptive_rate = 0.5  # Rate for adapting learning parameters

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        scores = np.array([func(ind) for ind in population])
        best_index = np.argmin(scores)
        best_solution = population[best_index].copy()
        evaluations = self.population_size

        while evaluations < self.budget:
            # Adaptive learning rate adjustment
            self.learning_rate = 0.05 + 0.45 * (1 - evaluations / self.budget)

            new_population = np.empty_like(population)
            for i in range(self.population_size):
                # Mutation via Gaussian noise
                noise = np.random.normal(0, self.mutation_strength, self.dim)
                offspring = population[i] + noise * self.learning_rate

                # Crossover with the best solution
                if np.random.rand() < self.crossover_rate:
                    cross_points = np.random.rand(self.dim) < self.adaptive_rate
                    offspring[cross_points] = best_solution[cross_points]

                # Ensure offspring are within bounds
                offspring = np.clip(offspring, lb, ub)
                new_population[i] = offspring

            # Evaluate new population
            new_scores = np.array([func(ind) for ind in new_population])
            evaluations += self.population_size

            # Select the best individuals from parent and offspring
            combined_population = np.vstack((population, new_population))
            combined_scores = np.hstack((scores, new_scores))
            best_indices = np.argsort(combined_scores)[:self.population_size]
            population = combined_population[best_indices]
            scores = combined_scores[best_indices]

            # Update best solution
            if scores[0] < func(best_solution):
                best_solution = population[0].copy()

        # Return the best solution found
        return best_solution, func(best_solution)