import numpy as np

class AdaptiveDEWithLocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim  # Typical choice is 10 times the dimension
        self.crossover_probability = 0.5
        self.mutation_factor = 0.5
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))

    def __call__(self, func):
        evaluations = 0
        best_solution = None
        best_fitness = float('inf')

        while evaluations < self.budget:
            new_population = np.copy(self.population)

            for i in range(self.population_size):
                # Mutation and crossover
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
                mutant_vector = np.clip(a + self.mutation_factor * (b - c), self.lower_bound, self.upper_bound)
                
                crossover = np.random.rand(self.dim) < self.crossover_probability
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True  # Ensure at least one dimension is crossed
                
                trial_vector = np.where(crossover, mutant_vector, self.population[i])
                
                # Local search using a simulated annealing-inspired approach
                accept_prob = np.random.rand()
                temperature = 1.0 - evaluations / self.budget
                local_search_vector = np.clip(trial_vector + np.random.normal(0, 0.1 * temperature, self.dim), self.lower_bound, self.upper_bound)

                trial_fitness = func(trial_vector)
                local_search_fitness = func(local_search_vector)
                evaluations += 2

                if local_search_fitness < trial_fitness and accept_prob < np.exp((trial_fitness - local_search_fitness) / temperature):
                    trial_vector = local_search_vector
                    trial_fitness = local_search_fitness

                # Selection
                if trial_fitness < func(self.population[i]):
                    new_population[i] = trial_vector
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_solution = trial_vector

                # Update adaptive parameters
                self.crossover_probability = 0.9 - 0.4 * (evaluations / self.budget)
                self.mutation_factor = 0.5 + 0.3 * np.sin(np.pi * evaluations / self.budget)

            self.population = new_population

        return best_solution