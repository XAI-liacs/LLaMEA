import numpy as np

class AdvancedHybridPSO_DE:
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
        self.dynamic_elite_factor = 0.1
        self.archive = []  # Adaptive memory archive

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        lower_bound, upper_bound = bounds[0], bounds[1]
        
        # Multi-swarm division
        num_swarms = 3
        sub_pop_size = self.population_size // num_swarms
        swarms = [self.particles[i*sub_pop_size:(i+1)*sub_pop_size] for i in range(num_swarms)]
        
        while self.fitness_evaluations < self.budget:
            # Dynamic elite size based on progress
            self.elite_size = max(1, int(self.population_size * self.dynamic_elite_factor))
            self.dynamic_elite_factor = 0.1 + 0.9 * (1 - self.fitness_evaluations / self.budget)

            for swarm in swarms:
                for i in range(sub_pop_size):
                    if self.fitness_evaluations >= self.budget:
                        break

                    # Evaluate particle fitness
                    fitness = func(swarm[i])
                    self.fitness_evaluations += 1

                    # Update personal and global bests
                    if fitness < self.personal_best_fitness[i]:
                        self.personal_best_fitness[i] = fitness
                        self.personal_best[i] = swarm[i].copy()

                    if fitness < self.global_best_fitness:
                        self.global_best_fitness = fitness
                        self.global_best = swarm[i].copy()

                    # Update archive with some probability
                    if np.random.rand() < 0.1 and len(self.archive) < self.population_size:
                        self.archive.append(swarm[i].copy())

            # Update particles using PSO and DE
            for swarm in swarms:
                for i in range(sub_pop_size):
                    if self.fitness_evaluations >= self.budget:
                        break

                    # PSO update with stochastic parameter adaptation
                    inertia_weight = 0.5 + 0.3 * np.random.rand()
                    cognitive_coeff = 1.5 + 0.5 * np.random.rand()
                    social_coeff = 1.5 + 0.5 * np.random.rand()
                    r1, r2 = np.random.rand(), np.random.rand()
                    cognitive_velocity = cognitive_coeff * r1 * (self.personal_best[i] - swarm[i])
                    social_velocity = social_coeff * r2 * (self.global_best - swarm[i])
                    self.velocities[i] = inertia_weight * self.velocities[i] + cognitive_velocity + social_velocity
                    swarm[i] += self.velocities[i]
                    swarm[i] = np.clip(swarm[i], lower_bound, upper_bound)

                    # DE mutation and crossover with dynamic elite selection
                    best_indices = np.argsort(self.personal_best_fitness)[:self.elite_size]
                    if len(self.archive) > 3:
                        archive_indices = np.random.choice(len(self.archive), 3, replace=False)
                        a, b, c = self.archive[archive_indices[0]], self.archive[archive_indices[1]], self.archive[archive_indices[2]]
                    else:
                        indices = np.random.choice(best_indices, 3, replace=False)
                        a, b, c = swarm[indices[0]], swarm[indices[1]], swarm[indices[2]]
                    F = 0.5 + 0.3 * np.random.rand()  # Adaptive mutation factor
                    mutant = np.clip(a + F * (b - c), lower_bound, upper_bound)
                    crossover_rate = 0.9
                    crossover_indices = np.random.rand(self.dim) < crossover_rate
                    trial = np.where(crossover_indices, mutant, swarm[i])

                    # Evaluate trial vector
                    trial_fitness = func(trial)
                    self.fitness_evaluations += 1

                    # Selection
                    if trial_fitness < self.personal_best_fitness[i]:
                        swarm[i] = trial.copy()
                        self.personal_best[i] = trial.copy()
                        self.personal_best_fitness[i] = trial_fitness
                        if trial_fitness < self.global_best_fitness:
                            self.global_best_fitness = trial_fitness
                            self.global_best = trial.copy()

        return self.global_best