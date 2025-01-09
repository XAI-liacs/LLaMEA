import numpy as np

class HybridDELocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.scale_factor = 0.8
        self.crossover_prob = 0.7
        self.local_search_steps = 5

    def __call__(self, func):
        np.random.seed(42)  # For reproducibility

        # Initialize the population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        budget_used = self.population_size

        while budget_used < self.budget:
            new_population = np.copy(population)

            # Differential Evolution Mutation and Crossover
            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = np.clip(population[a] + self.scale_factor * (population[b] - population[c]), 
                                 self.lower_bound, self.upper_bound)
                cross_points = np.random.rand(self.dim) < self.crossover_prob
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Evaluate trial vector
                trial_fitness = func(trial)
                budget_used += 1

                # Selection
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness

                # Check budget
                if budget_used >= self.budget:
                    break

            population = new_population

            # Local Search Phase
            for i in range(self.population_size):
                if budget_used >= self.budget:
                    break
                current = population[i]
                current_fitness = fitness[i]

                for _ in range(self.local_search_steps):
                    if budget_used >= self.budget:
                        break
                    step_size = np.random.uniform(-0.1, 0.1, self.dim)
                    neighbor = np.clip(current + step_size, self.lower_bound, self.upper_bound)
                    neighbor_fitness = func(neighbor)
                    budget_used += 1

                    if neighbor_fitness < current_fitness:
                        current = neighbor
                        current_fitness = neighbor_fitness

                population[i] = current
                fitness[i] = current_fitness

        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]