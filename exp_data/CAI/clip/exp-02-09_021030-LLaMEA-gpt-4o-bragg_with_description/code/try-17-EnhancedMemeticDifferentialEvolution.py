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
                mutant = np.clip(a + self.base_mutation_factor * (b - c), bounds.lb, bounds.ub)
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

            # Adjust mutation factor based on fitness diversity
            fitness_std = np.std(self.fitness)
            self.base_mutation_factor = 0.5 + 0.3 * (fitness_std / np.mean(self.fitness))

            if np.random.rand() < self.local_search_prob:
                self.adaptive_local_search(func, bounds, evaluations - last_improvement)
            if self.dynamic_population and evaluations - last_improvement > self.budget // 10:
                self.population_size = max(5, self.population_size // 2)

        return self.best_individual

    def adaptive_local_search(self, func, bounds, no_improvement_steps):
        for i in range(self.population_size):
            perturbation = np.random.uniform(-0.05, 0.05, self.dim)
            if no_improvement_steps > self.budget // 20:
                perturbation *= 2
            new_position = self.population[i] + perturbation
            new_position = np.clip(new_position, bounds.lb, bounds.ub)
            new_score = func(new_position)
            if new_score < self.fitness[i]:
                self.population[i] = new_position
                self.fitness[i] = new_score
                if new_score < func(self.best_individual):
                    self.best_individual = new_position