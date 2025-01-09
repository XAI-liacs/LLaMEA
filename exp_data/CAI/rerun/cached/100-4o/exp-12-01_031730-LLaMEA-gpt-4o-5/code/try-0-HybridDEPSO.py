import numpy as np

class HybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.velocity = np.random.uniform(-1.0, 1.0, (self.population_size, dim))
        self.best_positions = np.copy(self.population)
        self.best_fitness = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_fitness = np.inf
        self.eval_count = 0
    
    def __call__(self, func):
        while self.eval_count < self.budget:
            for i in range(self.population_size):
                if self.eval_count >= self.budget:
                    break
                # Evaluate fitness
                if self.fitness[i] == np.inf:
                    self.fitness[i] = func(self.population[i])
                    self.eval_count += 1

                # Update personal best
                if self.fitness[i] < self.best_fitness[i]:
                    self.best_fitness[i] = self.fitness[i]
                    self.best_positions[i] = self.population[i].copy()

                # Update global best
                if self.fitness[i] < self.global_best_fitness:
                    self.global_best_fitness = self.fitness[i]
                    self.global_best_position = self.population[i].copy()

            # Hybrid DE/PSO update
            for i in range(self.population_size):
                if self.eval_count >= self.budget:
                    break

                # Differential Evolution (DE)
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = self.population[indices]
                mutant_vector = a + 0.8 * (b - c)
                mutant_vector = np.clip(mutant_vector, -5.0, 5.0)

                # Crossover
                trial_vector = np.where(np.random.rand(self.dim) < 0.8, mutant_vector, self.population[i])
                trial_fitness = func(trial_vector)
                self.eval_count += 1

                # Selection
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial_vector
                    self.fitness[i] = trial_fitness

                # Update personal and global bests
                if self.fitness[i] < self.best_fitness[i]:
                    self.best_fitness[i] = self.fitness[i]
                    self.best_positions[i] = self.population[i].copy()

                if self.fitness[i] < self.global_best_fitness:
                    self.global_best_fitness = self.fitness[i]
                    self.global_best_position = self.population[i].copy()

            # Particle Swarm Optimization (PSO)
            for i in range(self.population_size):
                if self.eval_count >= self.budget:
                    break

                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                inertia_weight = 0.5 + np.random.rand() / 2

                # Update velocity
                self.velocity[i] = (inertia_weight * self.velocity[i] +
                                    2.0 * r1 * (self.best_positions[i] - self.population[i]) +
                                    2.0 * r2 * (self.global_best_position - self.population[i]))
                
                # Update position
                self.population[i] += self.velocity[i]
                self.population[i] = np.clip(self.population[i], -5.0, 5.0)

                # Evaluate new position
                self.fitness[i] = func(self.population[i])
                self.eval_count += 1

                # Update personal and global bests
                if self.fitness[i] < self.best_fitness[i]:
                    self.best_fitness[i] = self.fitness[i]
                    self.best_positions[i] = self.population[i].copy()

                if self.fitness[i] < self.global_best_fitness:
                    self.global_best_fitness = self.fitness[i]
                    self.global_best_position = self.population[i].copy()

        return self.global_best_position, self.global_best_fitness