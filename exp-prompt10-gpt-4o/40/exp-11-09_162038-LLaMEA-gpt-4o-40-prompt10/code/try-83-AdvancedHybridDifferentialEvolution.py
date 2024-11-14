import numpy as np

class AdvancedHybridDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(80, self.budget // 6)  # Further increased population size for enhanced diversity
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.F_base = 0.7  # Adjusting for more balanced exploration-exploitation
        self.CR_base = 0.8  # Adjusted to enhance convergence properties
        self.adaptation_rate = 0.1  # Increased for more dynamic parameter adjustment
        self.local_search_intensity = 0.2  # Further increased local search intensity
        self.elitism_rate = 0.1  # Inclusion of elitism for retaining top performers

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.population_size

        # Track the best solution found
        best_index = np.argmin(fitness)
        best_individual = population[best_index]
        best_fitness = fitness[best_index]

        while eval_count < self.budget:
            for i in range(self.population_size):
                # Dynamic F and CR
                F = np.clip(self.F_base + self.adaptation_rate * np.random.uniform(-1, 1), 0, 1)
                CR = np.clip(self.CR_base + self.adaptation_rate * np.random.uniform(-1, 1), 0, 1)

                # Choose mutation strategy based on a probability
                if np.random.rand() < 0.5:
                    indices = np.random.choice(self.population_size, 3, replace=False)
                    a, b, c = population[indices]
                    mutant = np.clip(a + F * (b - c), self.lower_bound, self.upper_bound)
                else:
                    indices = np.random.choice(self.population_size, 2, replace=False)
                    a, b = population[indices]
                    mutant = np.clip(best_individual + F * (a - b), self.lower_bound, self.upper_bound)

                trial = np.where(np.random.rand(self.dim) < CR, mutant, population[i])

                # Evaluate trial individual
                trial_fitness = func(trial)
                eval_count += 1

                # Selection and elitism
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_individual = trial
                        best_fitness = trial_fitness

                if eval_count >= self.budget:
                    break

            # Intensified cooperative local search around the best individual
            neighborhood_size = int(self.local_search_intensity * self.population_size)
            local_neighbors = best_individual + np.random.normal(0, 0.03, (neighborhood_size, self.dim))
            local_neighbors = np.clip(local_neighbors, self.lower_bound, self.upper_bound)
            local_fitness = np.array([func(ind) for ind in local_neighbors])
            eval_count += len(local_neighbors)

            # Update best if any local neighbor is better
            if np.min(local_fitness) < best_fitness:
                best_local_index = np.argmin(local_fitness)
                best_individual = local_neighbors[best_local_index]
                best_fitness = local_fitness[best_local_index]

            # Implement elitist learning
            elite_indices = np.argsort(fitness)[:int(self.elitism_rate * self.population_size)]
            elite_population = population[elite_indices]
            population[:len(elite_population)] = elite_population
            fitness[:len(elite_population)] = fitness[elite_indices]

            population[0] = best_individual
            fitness[0] = best_fitness

        return best_individual