import numpy as np

class HybridPSODERefined:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.c1_initial = 1.7  # Initial cognitive parameter
        self.c2_initial = 1.5  # Initial social parameter
        self.w_initial = 0.9   # Initial inertia weight
        self.w_final = 0.4     # Final inertia weight
        self.F = 0.6           # Differential evolution scale factor
        self.CR = 0.9          # Crossover probability
        self.velocity_clamp = (-1, 1)  # Velocity clamp to avoid explosion

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

    def update_particles(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
        cognitive_velocity = self.c1 * r1 * (self.personal_best - self.population)
        social_velocity = self.c2 * r2 * (self.global_best - self.population)
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

    def calculate_diversity(self):
        mean_pos = np.mean(self.population, axis=0)
        diversity = np.mean(np.linalg.norm(self.population - mean_pos, axis=1))
        return diversity

    def __call__(self, func):
        self.initialize_population(func.bounds)
        evaluations = 0
        diversity_threshold = 0.1 * np.linalg.norm(func.bounds.ub - func.bounds.lb)

        while evaluations < self.budget:
            self.evaluate_population(func)
            evaluations += self.population_size
            if evaluations >= self.budget:
                break

            # Calculate current diversity
            diversity = self.calculate_diversity()
            
            # Adaptive parameters based on diversity
            self.w = self.w_final + (self.w_initial - self.w_final) * np.exp(-diversity / diversity_threshold)
            self.c1 = self.c1_initial + (1.0 - diversity / diversity_threshold) * (2.5 - self.c1_initial)
            self.c2 = self.c2_initial + (diversity / diversity_threshold) * (2.5 - self.c2_initial)

            self.update_particles(func.bounds)
            evaluations += self.population_size
            if evaluations >= self.budget:
                break
            self.differential_evolution_step(func.bounds, func)
            evaluations += self.population_size

        return self.global_best, self.global_best_fitness