import numpy as np

class ImprovedHybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.particles = np.random.rand(self.population_size, dim)
        self.velocities = np.random.rand(self.population_size, dim) * 0.1
        self.personal_best = self.particles.copy()
        self.global_best = None
        self.personal_best_fitness = np.full(self.population_size, np.inf)
        self.global_best_fitness = np.inf
        self.fitness_evaluations = 0
        self.elite_size = max(1, self.population_size // 10)
        
    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        lower_bound, upper_bound = bounds[0], bounds[1]
        while self.fitness_evaluations < self.budget:
            for i in range(self.population_size):
                if self.fitness_evaluations >= self.budget:
                    break

                # Evaluate particle fitness
                fitness = func(self.particles[i])
                self.fitness_evaluations += 1

                # Update personal and global bests
                if fitness < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness
                    self.personal_best[i] = self.particles[i].copy()

                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best = self.particles[i].copy()

            # Update particles using PSO and DE
            for i in range(self.population_size):
                if self.fitness_evaluations >= self.budget:
                    break

                # PSO update with adaptive parameters
                inertia_weight = 0.4 + 0.5 * (1 - self.fitness_evaluations / self.budget)
                cognitive_coeff = 1.5 + 0.5 * np.random.rand()
                social_coeff = 1.5 + 0.5 * np.random.rand()
                r1, r2 = np.random.rand(), np.random.rand()
                cognitive_velocity = cognitive_coeff * r1 * (self.personal_best[i] - self.particles[i])
                social_velocity = social_coeff * r2 * (self.global_best - self.particles[i])
                self.velocities[i] = inertia_weight * self.velocities[i] + cognitive_velocity + social_velocity
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], lower_bound, upper_bound)

                # DE mutation and crossover with elite selection
                best_indices = np.argsort(self.personal_best_fitness)[:self.elite_size]
                indices = np.random.choice(best_indices, 3, replace=False)
                a, b, c = self.particles[indices[0]], self.particles[indices[1]], self.particles[indices[2]]
                F = 0.5 + 0.3 * np.random.rand()  # Adaptive mutation factor
                mutant = np.clip(a + F * (b - c), lower_bound, upper_bound)
                crossover_rate = 0.9
                crossover_indices = np.random.rand(self.dim) < crossover_rate
                trial = np.where(crossover_indices, mutant, self.particles[i])

                # Evaluate trial vector
                trial_fitness = func(trial)
                self.fitness_evaluations += 1

                # Selection
                if trial_fitness < self.personal_best_fitness[i]:
                    self.particles[i] = trial.copy()
                    self.personal_best[i] = trial.copy()
                    self.personal_best_fitness[i] = trial_fitness
                    if trial_fitness < self.global_best_fitness:
                        self.global_best_fitness = trial_fitness
                        self.global_best = trial.copy()

        return self.global_best