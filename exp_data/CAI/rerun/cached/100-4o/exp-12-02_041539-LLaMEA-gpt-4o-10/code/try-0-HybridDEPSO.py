import numpy as np

class HybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = 20
        self.cr = 0.9  # Crossover probability for DE
        self.f = 0.8   # Differential weight for DE
        self.w = 0.5   # Inertia weight for PSO
        self.c1 = 1.5  # Cognitive coefficient for PSO
        self.c2 = 1.5  # Social coefficient for PSO
        self.velocities = np.zeros((self.population_size, self.dim))
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.population)
        self.global_best_position = None
        self.evaluations = 0

    def __call__(self, func):
        fitness = np.apply_along_axis(func, 1, self.population)
        self.evaluations += self.population_size
        self.personal_best_fitness = np.copy(fitness)
        self.global_best_position = self.population[np.argmin(fitness)]
        
        while self.evaluations < self.budget:
            # Differential Evolution (DE) step
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = self.population[indices]
                mutant = np.clip(a + self.f * (b - c), self.bounds[0], self.bounds[1])
                cross_points = np.random.rand(self.dim) < self.cr
                trial = np.where(cross_points, mutant, self.population[i])
                trial_fitness = func(trial)
                self.evaluations += 1

                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    self.population[i] = trial
                    if trial_fitness < self.personal_best_fitness[i]:
                        self.personal_best_fitness[i] = trial_fitness
                        self.personal_best_positions[i] = trial
                        if trial_fitness < func(self.global_best_position):
                            self.global_best_position = trial

            # Particle Swarm Optimization (PSO) step
            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                self.velocities[i] = (self.w * self.velocities[i]
                                      + self.c1 * r1 * (self.personal_best_positions[i] - self.population[i])
                                      + self.c2 * r2 * (self.global_best_position - self.population[i]))
                new_position = np.clip(self.population[i] + self.velocities[i], self.bounds[0], self.bounds[1])
                new_fitness = func(new_position)
                self.evaluations += 1

                if new_fitness < fitness[i]:
                    fitness[i] = new_fitness
                    self.population[i] = new_position
                    if new_fitness < self.personal_best_fitness[i]:
                        self.personal_best_fitness[i] = new_fitness
                        self.personal_best_positions[i] = new_position
                        if new_fitness < func(self.global_best_position):
                            self.global_best_position = new_position

        return self.global_best_position