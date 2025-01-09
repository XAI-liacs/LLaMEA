import numpy as np

class HPSO_ADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20
        self.inertia_weight = 0.7
        self.cognitive_constant = 1.5
        self.social_constant = 1.5
        self.F = 0.8  # DE mutation factor
        self.CR = 0.9  # DE crossover probability
        self.population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )
        self.velocities = np.random.uniform(
            -1, 1, (self.population_size, self.dim)
        )
        self.personal_best = self.population.copy()
        self.global_best = self.population[np.argmin([float('inf')] * self.population_size)]
        self.fitness = np.array([float('inf')] * self.population_size)
        self.best_fitness = float('inf')
        self.evaluations = 0

    def __call__(self, func):
        while self.evaluations < self.budget:
            for i in range(self.population_size):
                # PSO Update
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                self.velocities[i] = (
                    self.inertia_weight * self.velocities[i]
                    + self.cognitive_constant * r1 * (self.personal_best[i] - self.population[i])
                    + self.social_constant * r2 * (self.global_best - self.population[i])
                )
                self.population[i] += self.velocities[i]
                self.population[i] = np.clip(self.population[i], self.lower_bound, self.upper_bound)
                
                # Evaluate fitness
                fitness = func(self.population[i])
                self.evaluations += 1
                if fitness < self.fitness[i]:
                    self.fitness[i] = fitness
                    self.personal_best[i] = self.population[i].copy()
                    
                    # Update global best
                    if fitness < self.best_fitness:
                        self.best_fitness = fitness
                        self.global_best = self.population[i].copy()
                
                if self.evaluations >= self.budget:
                    break
            
            # Adaptive Differential Evolution Update
            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
                self.F = 0.5 + 0.3 * np.cos(2 * np.pi * self.evaluations / self.budget)  # Dynamic adaptation
                mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)
                trial = np.random.rand(self.dim) < self.CR
                offspring = np.where(trial, mutant, self.population[i])
                
                # Evaluate fitness of offspring
                fitness = func(offspring)
                self.evaluations += 1
                
                if fitness < self.fitness[i]:
                    self.population[i] = offspring
                    self.fitness[i] = fitness
                    if fitness < self.best_fitness:
                        self.best_fitness = fitness
                        self.global_best = offspring
                
                if self.evaluations >= self.budget:
                    break
        
        return self.global_best