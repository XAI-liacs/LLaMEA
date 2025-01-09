import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(p) for p in population])
        self.budget -= self.population_size
        
        F = np.random.uniform(0.4, 0.9, self.population_size)  # Adaptive mutation factor
        CR = np.random.uniform(0.2, 0.9, self.population_size)  # Adaptive crossover rate
        chaotic_sequence = np.random.rand(self.dim)
        
        while self.budget > 0:
            for i in range(self.population_size):
                if self.budget <= 0:
                    break
                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = population[indices]
                F_i = F[i] + 0.01 * chaotic_sequence
                chaotic_sequence = np.mod(chaotic_sequence * 3.57 * (1 - chaotic_sequence) + 0.01, 1)
                
                mutant = np.clip(x1 + F_i * (x2 - x3), lb, ub)
                crossover_mask = np.random.rand(self.dim) < CR[i]
                trial = np.where(crossover_mask, mutant, population[i])
                
                trial_fitness = func(trial)
                self.budget -= 1
                
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                
                if fitness[i] < self.f_opt:
                    self.f_opt = fitness[i]
                    self.x_opt = population[i]
        
        return self.f_opt, self.x_opt