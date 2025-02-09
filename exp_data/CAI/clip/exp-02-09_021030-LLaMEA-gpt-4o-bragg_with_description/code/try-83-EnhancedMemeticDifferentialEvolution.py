import numpy as np

class EnhancedMemeticDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.base_mutation_factor = 0.8
        self.base_crossover_rate = 0.9
        self.local_search_prob = 0.2
        self.dynamic_population = True
        self.inertia_weight = 0.9  # Adaptive inertia weight
        self.eta = 0.1

    def initialize_population(self, bounds):
        self.population = np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim))
        self.fitness = np.array([float('inf')] * self.population_size)
        self.best_individual = None
        self.velocity = np.zeros((self.population_size, self.dim))  # Initialize velocity for individuals

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
                mutant = np.clip(a + self.base_mutation_factor * (b - c), bounds.lb, bounds.ub)
                crossover_rate = self.base_crossover_rate * (1.0 - np.var(self.fitness) / np.mean(self.fitness))
                crossover_rate = min(max(crossover_rate, 0.1), 0.9)
                trial = np.where(np.random.rand(self.dim) < crossover_rate, mutant, self.population[i])

                self.velocity[i] = self.inertia_weight * self.velocity[i] + self.eta * (mutant - self.population[i])
                trial += self.velocity[i]  # Apply velocity to trial
                trial = np.clip(trial, bounds.lb, bounds.ub)
                
                trial_score = func(trial)
                evaluations += 1

                if trial_score < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_score
                    if trial_score < best_score:
                        best_score = trial_score
                        self.best_individual = trial
                        last_improvement = evaluations

            self.base_mutation_factor = np.clip(0.5 + 0.3 * (np.std(self.fitness) / np.mean(self.fitness)), 0.5, 1.0)
            self.local_search_prob = 0.2 + 0.1 * (best_score / np.min(self.fitness))
            self.inertia_weight = max(0.4, self.inertia_weight * 0.99)  # Reduce inertia weight over time

            if np.random.rand() < self.local_search_prob:
                self.chaotic_local_search(func, bounds)
            if self.dynamic_population and evaluations - last_improvement > self.budget // 10:
                self.population_size = max(5, self.population_size // 2)

            self.maintain_diversity()

        return self.best_individual

    def chaotic_local_search(self, func, bounds):
        for i in range(self.population_size):
            perturbation = np.random.uniform(-0.2, 0.2, self.dim) * np.sin(np.pi * np.random.rand(self.dim))  # Chaotic
            new_position = np.clip(self.population[i] + perturbation, bounds.lb, bounds.ub)
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
                self.population[i] += np.random.uniform(-0.15, 0.15, self.dim)