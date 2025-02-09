import numpy as np

class MemeticDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.local_search_prob = 0.3  # Increased local search probability
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
                mutant = np.clip(a + self.mutation_factor * (b - c), bounds.lb, bounds.ub)
                trial = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant, self.population[i])

                trial_score = func(trial)
                evaluations += 1

                if trial_score < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_score
                    if trial_score < best_score:
                        best_score = trial_score
                        self.best_individual = trial
                        last_improvement = evaluations

            self.mutation_factor = 0.5 + 0.5 * np.random.rand()

            if np.random.rand() < self.local_search_prob:
                self.multi_phase_local_search(func, bounds, evaluations - last_improvement)  # Modified to multi-phase search
            if self.dynamic_population and evaluations - last_improvement > self.budget // 10:
                self.population_size = max(5, int(self.population_size * 0.7))  # Adaptive reduction factor

        return self.best_individual

    def multi_phase_local_search(self, func, bounds, no_improvement_steps):
        for i in range(self.population_size):
            step_size = 0.01 if no_improvement_steps < self.budget // 20 else 0.1
            perturbation = np.random.uniform(-step_size, step_size, self.dim)
            new_position = self.population[i] + perturbation
            new_position = np.clip(new_position, bounds.lb, bounds.ub)
            new_score = func(new_position)
            if new_score < self.fitness[i]:
                self.population[i] = new_position
                self.fitness[i] = new_score
                if new_score < func(self.best_individual):
                    self.best_individual = new_position