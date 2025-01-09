import numpy as np

class AdaptiveModeSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.c1 = 1.5  # Cognitive parameter
        self.c2 = 1.5  # Social parameter
        self.w_min = 0.4
        self.w_max = 0.9
        self.F_base = 0.5
        self.CR = 0.9
        self.velocity_clamp = (-1, 1)

    def diversity_metric(self, population):
        return np.mean(np.std(population, axis=0))

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

    def update_particles(self, bounds, w):
        lb, ub = bounds.lb, bounds.ub
        r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
        cognitive_velocity = self.c1 * r1 * (self.personal_best - self.population)
        social_velocity = self.c2 * r2 * (self.global_best - self.population)
        self.velocities = w * self.velocities + cognitive_velocity + social_velocity
        self.velocities = np.clip(self.velocities, *self.velocity_clamp)
        self.population = self.population + self.velocities
        self.population = np.clip(self.population, lb, ub)

    def adaptive_differential_evolution(self, bounds, diversity, func):
        lb, ub = bounds.lb, bounds.ub
        F = self.F_base + 0.5 * diversity  # Adaptive scale factor based on diversity
        for i in range(self.population_size):
            idxs = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
            mutant_vector = np.clip(a + F * (b - c), lb, ub)
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

    def __call__(self, func):
        self.initialize_population(func.bounds)
        evaluations = 0

        while evaluations < self.budget:
            self.evaluate_population(func)
            evaluations += self.population_size
            if evaluations >= self.budget:
                break
            
            diversity = self.diversity_metric(self.population)
            w = self.w_min + (self.w_max - self.w_min) * (1 - diversity)
            self.update_particles(func.bounds, w)
            evaluations += self.population_size
            if evaluations >= self.budget:
                break
            
            self.adaptive_differential_evolution(func.bounds, diversity, func)
            evaluations += self.population_size

        return self.global_best, self.global_best_fitness