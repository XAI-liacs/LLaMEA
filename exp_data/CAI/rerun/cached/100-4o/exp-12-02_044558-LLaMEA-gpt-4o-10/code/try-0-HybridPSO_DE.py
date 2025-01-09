import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 40
        self.c1 = 2.0
        self.c2 = 2.0
        self.w = 0.5
        self.F = 0.5
        self.CR = 0.9
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best = np.copy(self.population)
        self.personal_best_fitness = np.full(self.population_size, np.inf)
        self.global_best = None
        self.global_best_fitness = np.inf
        self.eval_count = 0

    def __call__(self, func):
        while self.eval_count < self.budget:
            # Evaluate fitness
            fitness = np.array([func(ind) for ind in self.population])
            self.eval_count += self.population_size
            
            # Update personal and global bests
            for i in range(self.population_size):
                if fitness[i] < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness[i]
                    self.personal_best[i] = self.population[i]
            
            if np.min(fitness) < self.global_best_fitness:
                self.global_best_fitness = np.min(fitness)
                self.global_best = self.population[np.argmin(fitness)]
            
            # PSO update
            r1, r2 = np.random.uniform(size=(2, self.population_size, self.dim))
            self.velocities = (
                self.w * self.velocities +
                self.c1 * r1 * (self.personal_best - self.population) +
                self.c2 * r2 * (self.global_best - self.population)
            )
            self.population += self.velocities
            self.population = np.clip(self.population, self.lower_bound, self.upper_bound)
            
            # DE mutation and crossover
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, self.population[i])
                trial_fitness = func(trial)
                self.eval_count += 1
                
                if trial_fitness < fitness[i]:
                    self.population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < self.personal_best_fitness[i]:
                        self.personal_best_fitness[i] = trial_fitness
                        self.personal_best[i] = trial

                if trial_fitness < self.global_best_fitness:
                    self.global_best_fitness = trial_fitness
                    self.global_best = trial

                if self.eval_count >= self.budget:
                    break

        return self.global_best, self.global_best_fitness