import numpy as np

class EnhancedMemeticDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30  # Adjusted population size for better exploration
        self.base_mutation_factor = 0.7  # Adjusted mutation factor for stability
        self.base_crossover_rate = 0.85  # Slightly reduced crossover rate
        self.local_search_prob = 0.25  # Increased probability of local search
        self.dynamic_population = True

    def initialize_population(self, bounds):
        self.population = np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim))
        self.fitness = np.array([float('inf')] * self.population_size)
        self.best_individual = None

    def __call__(self, func):
        bounds = func.bounds
        self.initialize_population(bounds)
        evaluations = 0
        best_score = float('inf')
        last_improvement = 0

        while evaluations < self.budget:
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = self.population[indices]
                chaotic_factor = 0.5 * np.sin(evaluations) + 0.5  # Chaotic map-based factor
                mutant = np.clip(a + chaotic_factor * (b - c), bounds.lb, bounds.ub)
                crossover_rate = self.base_crossover_rate * (1.0 - np.var(self.fitness) / np.max(self.fitness))
                trial = np.where(np.random.rand(self.dim) < crossover_rate, mutant, self.population[i])

                trial_score = func(trial)
                evaluations += 1

                if trial_score < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_score
                    if trial_score < best_score:
                        best_score = trial_score
                        self.best_individual = trial
                        last_improvement = evaluations

            # Adjust mutation factor dynamically
            fitness_std = np.std(self.fitness)
            self.base_mutation_factor = np.clip(0.6 + 0.2 * (fitness_std / np.mean(self.fitness)), 0.4, 0.9)

            if np.random.rand() < self.local_search_prob:
                self.adaptive_local_search(func, bounds, evaluations - last_improvement)
            if self.dynamic_population and evaluations - last_improvement > self.budget // 15:
                self.population_size = max(5, self.population_size // 2)

            self.maintain_diversity()

        return self.best_individual

    def adaptive_local_search(self, func, bounds, no_improvement_steps):
        for i in range(self.population_size):
            perturbation = np.random.uniform(-0.05, 0.05, self.dim)
            if no_improvement_steps > self.budget // 25:
                perturbation *= 3  # Increased perturbation when stuck
            new_position = self.population[i] + perturbation
            new_position = np.clip(new_position, bounds.lb, bounds.ub)
            new_score = func(new_position)
            if new_score < self.fitness[i]:
                self.population[i] = new_position
                self.fitness[i] = new_score
                if new_score < func(self.best_individual):
                    self.best_individual = new_position

    def maintain_diversity(self):
        distances = np.sum((self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]) ** 2, axis=2)
        np.fill_diagonal(distances, np.inf)
        closest_indices = np.argmin(distances, axis=1)
        for i in range(self.population_size):
            if np.all(self.fitness[i] == self.fitness[closest_indices[i]]):
                self.population[i] += np.random.uniform(-0.08, 0.08, self.dim)  # Adjusted diversity maintenance