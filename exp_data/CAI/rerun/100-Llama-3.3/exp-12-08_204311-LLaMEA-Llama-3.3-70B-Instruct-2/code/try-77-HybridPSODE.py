import numpy as np

class HybridPSODE:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 50
        self.particles = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.velocities = np.zeros((self.population_size, self.dim))
        self.personal_best = self.particles.copy()
        self.personal_best_fitness = np.ones(self.population_size) * np.Inf
        self.global_best = np.zeros(self.dim)
        self.global_best_fitness = np.Inf
        self.cr = 0.5  # crossover rate for DE
        self.f = 0.5  # scaling factor for DE

    def __call__(self, func):
        for i in range(self.budget):
            # Evaluate the fitness of each particle
            fitness = np.array([func(x) for x in self.particles])
            
            # Update personal best
            for j in range(self.population_size):
                if fitness[j] < self.personal_best_fitness[j]:
                    self.personal_best[j] = self.particles[j]
                    self.personal_best_fitness[j] = fitness[j]
                    
            # Update global best
            idx = np.argmin(fitness)
            if fitness[idx] < self.global_best_fitness:
                self.global_best = self.particles[idx]
                self.global_best_fitness = fitness[idx]
                self.f_opt = self.global_best_fitness
                self.x_opt = self.global_best
                
            # Update velocities using PSO with a slightly increased inertia coefficient
            for j in range(self.population_size):
                self.velocities[j] = 0.74 * self.velocities[j] + 1.494 * np.random.uniform(0, 1, self.dim) * (self.personal_best[j] - self.particles[j]) + 1.494 * np.random.uniform(0, 1, self.dim) * (self.global_best - self.particles[j])
                
            # Update positions using DE
            for j in range(self.population_size):
                r1, r2, r3 = np.random.choice(self.population_size, 3, replace=False)
                mutant = self.particles[r1] + self.f * (self.particles[r2] - self.particles[r3])
                trial = np.where(np.random.uniform(0, 1, self.dim) < self.cr, mutant, self.particles[j])
                trial = np.clip(trial, -5.0, 5.0)
                trial_fitness = func(trial)
                if trial_fitness < fitness[j]:
                    self.particles[j] = trial
                    fitness[j] = trial_fitness
                    
        return self.f_opt, self.x_opt