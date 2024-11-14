import numpy as np

class EnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 20 * dim  # Slightly increased initial population
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        population_size = self.initial_population_size
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        evaluations = population_size
        elite_fraction = 0.15  # Slightly reduced to foster diversity
        local_search_prob = 0.1

        while evaluations < self.budget:
            progress_ratio = evaluations / self.budget
            mutation_factor = 0.5 + 0.5 * np.random.rand()  # Enhanced dynamic mutation range
            crossover_rate = 0.5 + 0.4 * np.random.rand()  # More dynamic range for crossover

            new_population = np.copy(population)

            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + mutation_factor * (b - c), self.lower_bound, self.upper_bound)

                trial = np.array([
                    mutant[j] if np.random.rand() < crossover_rate or j == np.random.randint(self.dim) else population[i][j]
                    for j in range(self.dim)
                ])

                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

                if evaluations >= self.budget:
                    break

            if evaluations < self.budget * 0.6:  # Early local search to aid exploration
                local_search_indices = np.random.choice(population_size, int(local_search_prob * population_size), replace=False)
                for idx in local_search_indices:
                    candidate = new_population[idx] + np.random.normal(0, 0.1, self.dim)
                    candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
                    candidate_fitness = func(candidate)
                    evaluations += 1
                    if candidate_fitness < fitness[idx]:
                        new_population[idx] = candidate
                        fitness[idx] = candidate_fitness

                    if candidate_fitness < best_fitness:
                        best_solution = candidate
                        best_fitness = candidate_fitness

            if evaluations < self.budget and evaluations > self.budget * 0.4:  # Slightly earlier elitism
                elite_size = max(1, int(elite_fraction * population_size))
                sorted_indices = np.argsort(fitness)
                elite_population = population[sorted_indices[:elite_size]]
                new_population_size = max(4, int(self.initial_population_size * (1 - progress_ratio)))
                new_population = np.vstack((new_population[sorted_indices[:new_population_size - elite_size]], elite_population))
                fitness = fitness[sorted_indices[:new_population_size]]
                population_size = new_population_size

            population = new_population

        return best_solution