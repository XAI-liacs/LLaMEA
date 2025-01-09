import numpy as np

class AdaptiveEvolutionaryStrategy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.sigma = 0.3
        self.success_threshold = 0.2
        self.success_rate = 0.0
        self.success_count = 0
        self.evaluations = 0
        self.population_size = 15
        self.step_size_adjustment = 1.2

    def __call__(self, func):
        # Initialize the population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_index = np.argmin(fitness)
        best_individual = population[best_index]
        best_fitness = fitness[best_index]
        
        while self.evaluations < self.budget:
            offspring = np.array([self.mutate(ind) for ind in population])
            offspring_fitness = np.array([func(ind) for ind in offspring])
            
            new_population = []
            new_fitness = []

            # Select the best individuals from current and offspring
            combined_population = np.vstack((population, offspring))
            combined_fitness = np.hstack((fitness, offspring_fitness))
            indices = np.argsort(combined_fitness)[:self.population_size]
            
            for idx in indices:
                new_population.append(combined_population[idx])
                new_fitness.append(combined_fitness[idx])

            # Update success rate and adjust sigma
            successful_mutations = np.sum(offspring_fitness < fitness)
            self.success_rate = successful_mutations / self.population_size
            if self.success_rate > self.success_threshold:
                self.sigma *= self.step_size_adjustment
            else:
                self.sigma /= self.step_size_adjustment
            
            population = np.array(new_population)
            fitness = np.array(new_fitness)
            
            # Update best found solution
            current_best_index = np.argmin(fitness)
            if fitness[current_best_index] < best_fitness:
                best_fitness = fitness[current_best_index]
                best_individual = population[current_best_index]
            
            self.evaluations += self.population_size * 2

        return best_individual

    def mutate(self, individual):
        mutation = np.random.normal(0, self.sigma, self.dim)
        new_individual = np.clip(individual + mutation, self.lower_bound, self.upper_bound)
        return new_individual