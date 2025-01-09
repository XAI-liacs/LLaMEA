import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.pop = np.random.uniform(-5, 5, (self.population_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, dim))
        self.p_best = self.pop.copy()
        self.g_best = self.pop[np.random.choice(self.population_size)]
        self.fitness = np.full(self.population_size, np.inf)
        self.fitness_p_best = np.full(self.population_size, np.inf)
        self.fitness_g_best = np.inf
        self.w = 0.5
        self.c1 = 1.5
        self.c2 = 1.5
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9

    def __call__(self, func):
        evals = 0
        while evals < self.budget:
            for i in range(self.population_size):
                # Evaluate fitness
                self.fitness[i] = func(self.pop[i])
                evals += 1

                # Update personal best
                if self.fitness[i] < self.fitness_p_best[i]:
                    self.p_best[i] = self.pop[i]
                    self.fitness_p_best[i] = self.fitness[i]

                # Update global best
                if self.fitness[i] < self.fitness_g_best:
                    self.g_best = self.pop[i]
                    self.fitness_g_best = self.fitness[i]

            # PSO Update - velocity and position
            for i in range(self.population_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                cognitive = self.c1 * r1 * (self.p_best[i] - self.pop[i])
                social = self.c2 * r2 * (self.g_best - self.pop[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive + social
                self.pop[i] += self.velocities[i]
                self.pop[i] = np.clip(self.pop[i], -5, 5)

            # DE Update - mutation and crossover
            for i in range(self.population_size):
                if evals >= self.budget:
                    break

                candidates = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.pop[np.random.choice(candidates, 3, replace=False)]

                mutant = np.clip(a + self.mutation_factor * (b - c), -5, 5)
                cross_points = np.random.rand(self.dim) < self.crossover_rate
                trial = np.where(cross_points, mutant, self.pop[i])

                trial_fitness = func(trial)
                evals += 1
                
                if trial_fitness < self.fitness[i]:
                    self.pop[i] = trial
                    self.fitness[i] = trial_fitness
                    if trial_fitness < self.fitness_p_best[i]:
                        self.p_best[i] = trial
                        self.fitness_p_best[i] = trial_fitness
                    if trial_fitness < self.fitness_g_best:
                        self.g_best = trial
                        self.fitness_g_best = trial_fitness