import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 30
        self.population_size = 50
        self.particles = np.random.uniform(-5.0, 5.0, (self.swarm_size, self.dim))
        self.velocities = np.random.uniform(-1.0, 1.0, (self.swarm_size, self.dim))
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.personal_best = self.particles.copy()
        self.personal_best_value = np.full(self.swarm_size, np.inf)
        self.global_best = None
        self.global_best_value = np.inf
        self.alpha = 0.5
        self.beta = 0.5
        self.w = 0.5
        self.c1 = 1.5
        self.c2 = 1.5
        self.f = 0.5
        self.cr = 0.9
    
    def __call__(self, func):
        evaluations = 0
        
        while evaluations < self.budget:
            # Evaluate the swarm
            for i in range(self.swarm_size):
                if evaluations >= self.budget:
                    break
                fitness = func(self.particles[i])
                evaluations += 1
                if fitness < self.personal_best_value[i]:
                    self.personal_best_value[i] = fitness
                    self.personal_best[i] = self.particles[i].copy()
                if fitness < self.global_best_value:
                    self.global_best_value = fitness
                    self.global_best = self.particles[i].copy()
            
            # PSO Update
            for i in range(self.swarm_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                cognitive = self.c1 * r1 * (self.personal_best[i] - self.particles[i])
                social = self.c2 * r2 * (self.global_best - self.particles[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive + social
                self.particles[i] = self.particles[i] + self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], -5.0, 5.0)
            
            # Evaluate the population
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                fitness = func(self.population[i])
                evaluations += 1
                if fitness < self.global_best_value:
                    self.global_best_value = fitness
                    self.global_best = self.population[i].copy()
            
            # DE Update
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                candidates = np.random.choice(np.delete(np.arange(self.population_size), i), 3, replace=False)
                x1, x2, x3 = self.population[candidates]
                mutant = x1 + self.f * (x2 - x3)
                mutant = np.clip(mutant, -5.0, 5.0)
                trial = np.where(np.random.rand(self.dim) < self.cr, mutant, self.population[i])
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness:
                    self.population[i] = trial
        
        return self.global_best, self.global_best_value