import numpy as np

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = max(10, 5 * dim)  # Ensure a reasonable population size
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.temperature = 1000.0
        self.cooling_rate = 0.99
        self.eval_count = 0

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        self.eval_count += self.population_size

        while self.eval_count < self.budget:
            new_population = np.copy(population)
            for i in range(self.population_size):
                if self.eval_count >= self.budget:
                    break
                # Mutation
                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant_vector = np.clip(x1 + self.mutation_factor * (x2 - x3), self.lower_bound, self.upper_bound)
                
                # Crossover
                trial_vector = np.copy(population[i])
                for j in range(self.dim):
                    if np.random.rand() < self.crossover_rate:
                        trial_vector[j] = mutant_vector[j]
                
                # Selection
                trial_fitness = func(trial_vector)
                self.eval_count += 1
                if trial_fitness < fitness[i]:
                    new_population[i] = trial_vector
                    fitness[i] = trial_fitness

            # Simulated Annealing
            for i in range(self.population_size):
                if self.eval_count >= self.budget:
                    break
                candidate = new_population[i] + np.random.normal(0, 1, self.dim) * (self.temperature / 1000.0)
                candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
                candidate_fitness = func(candidate)
                self.eval_count += 1
                delta_fitness = candidate_fitness - fitness[i]
                acceptance_prob = np.exp(-delta_fitness / self.temperature)
                if candidate_fitness < fitness[i] or np.random.rand() < acceptance_prob:
                    new_population[i] = candidate
                    fitness[i] = candidate_fitness

            # Update population and temperature
            population[:] = new_population
            self.temperature *= self.cooling_rate

        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]