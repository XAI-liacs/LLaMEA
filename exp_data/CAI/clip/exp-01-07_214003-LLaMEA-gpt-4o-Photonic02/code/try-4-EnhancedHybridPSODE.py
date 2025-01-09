import numpy as np

class EnhancedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.c1 = 1.5  # Cognitive parameter
        self.c2 = 1.5  # Social parameter
        self.w_max = 0.9
        self.w_min = 0.4
        self.F = 0.5   # Differential evolution scale factor
        self.CR = 0.9  # Crossover probability
        self.velocity_clamp = (-1, 1)  # Velocity clamp to avoid explosion
        self.local_search_frequency = 5

    def initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best = self.population.copy()
        self.personal_best_fitness = np.full(self.population_size, np.inf)
        self.global_best = None
        self.global_best_fitness = np.inf

    def evaluate_population(self, func):
        for i in range(self.population_size):
            fitness = func(self.population[i])
            if fitness < self.personal_best_fitness[i]:
                self.personal_best_fitness[i] = fitness
                self.personal_best[i] = self.population[i].copy()
            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best = self.population[i].copy()

    def update_particles(self, bounds, iteration, max_iterations):
        lb, ub = bounds.lb, bounds.ub
        r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
        cognitive_velocity = self.c1 * r1 * (self.personal_best - self.population)
        social_velocity = self.c2 * r2 * (self.global_best - self.population)
        self.w = self.w_max - ((self.w_max - self.w_min) * (iteration / max_iterations))  # Annealing-inspired inertia
        self.velocities = self.w * self.velocities + cognitive_velocity + social_velocity
        self.velocities = np.clip(self.velocities, *self.velocity_clamp)
        self.population = self.population + self.velocities
        self.population = np.clip(self.population, lb, ub)

    def differential_evolution_step(self, bounds, func):
        lb, ub = bounds.lb, bounds.ub
        for i in range(self.population_size):
            idxs = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
            mutant_vector = np.clip(a + self.F * (b - c), lb, ub)
            crossover = np.random.rand(self.dim) < self.CR
            trial_vector = np.where(crossover, mutant_vector, self.population[i])
            trial_fitness = func(trial_vector)
            if trial_fitness < self.personal_best_fitness[i]:
                self.population[i] = trial_vector
                self.personal_best_fitness[i] = trial_fitness
                self.personal_best[i] = trial_vector
                if trial_fitness < self.global_best_fitness:
                    self.global_best_fitness = trial_fitness
                    self.global_best = trial_vector

    def local_search(self, func):
        for i in range(self.population_size):
            candidate = self.population[i] + np.random.normal(0, 0.1, self.dim)
            candidate_fitness = func(candidate)
            if candidate_fitness < self.personal_best_fitness[i]:
                self.population[i] = candidate
                self.personal_best_fitness[i] = candidate_fitness
                self.personal_best[i] = candidate
                if candidate_fitness < self.global_best_fitness:
                    self.global_best_fitness = candidate_fitness
                    self.global_best = candidate

    def __call__(self, func):
        self.initialize_population(func.bounds)
        evaluations = 0
        max_iterations = int(self.budget / self.population_size)
        iteration = 0

        while evaluations < self.budget:
            self.evaluate_population(func)
            evaluations += self.population_size
            if evaluations >= self.budget:
                break
            self.update_particles(func.bounds, iteration, max_iterations)
            evaluations += self.population_size
            if evaluations >= self.budget:
                break
            self.differential_evolution_step(func.bounds, func)
            evaluations += self.population_size
            if evaluations >= self.budget:
                break
            if iteration % self.local_search_frequency == 0:
                self.local_search(func)
            iteration += 1

        return self.global_best, self.global_best_fitness