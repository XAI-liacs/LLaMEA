import numpy as np

class EnhancedHybridDE_SLS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20 + int(0.4 * dim)  # Adjusted population size for scalability
        self.prob_crossover = 0.85  # Balanced crossover probability
        self.F = 0.9  # Slightly higher differential weight for exploration
        self.current_evaluations = 0
        self.temperature = 2.0  # Higher initial temperature for broader search
        self.cooling_rate = 0.95  # More gradual cooling for sustained search ability
        self.diversity_factor = 0.25  # Enhanced diversity factor
        self.local_search_probability = 0.3  # Probability for local search

    def __call__(self, func):
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )
        fitness = np.array([func(ind) for ind in population])
        self.current_evaluations += self.population_size
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        while self.current_evaluations < self.budget:
            for i in range(self.population_size):
                if self.current_evaluations >= self.budget:
                    break

                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = indices
                while a == i or b == i or c == i:
                    a, b, c = np.random.choice(self.population_size, 3, replace=False)

                mutant = population[a] + self.F * (population[b] - population[c])
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                
                if np.random.rand() < self.diversity_factor:
                    direction = np.random.randn(self.dim)
                    direction /= np.linalg.norm(direction)
                    mutant = population[i] + direction * (self.upper_bound - self.lower_bound) * 0.15
                    mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                crossover_mask = np.random.rand(self.dim) < self.prob_crossover
                trial = np.where(crossover_mask, mutant, population[i])

                if np.random.rand() < self.local_search_probability:
                    trial = trial + np.random.normal(0, 0.1, self.dim)
                    trial = np.clip(trial, self.lower_bound, self.upper_bound)

                trial_fitness = func(trial)
                self.current_evaluations += 1
                delta_e = trial_fitness - fitness[i]
                acceptance_probability = np.exp(-delta_e / (self.temperature + 1e-9))

                self.temperature *= self.cooling_rate

                if trial_fitness < fitness[i] or np.random.rand() < acceptance_probability:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

        return best_solution