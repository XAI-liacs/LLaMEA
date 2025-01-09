import numpy as np

class DEPSOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.velocities = np.random.uniform(-1, 1, (self.pop_size, dim))
        self.p_best = self.population.copy()
        self.p_best_fitness = np.full(self.pop_size, np.inf)
        self.g_best = None
        self.g_best_fitness = np.inf
    
    def __call__(self, func):
        evaluations = 0
        adapt_rate = 0.9
        
        while evaluations < self.budget:
            for i in range(self.pop_size):
                if self.fitness[i] == np.inf:
                    self.fitness[i] = func(self.population[i])
                    evaluations += 1
                    if evaluations >= self.budget:
                        break
            
            for i in range(self.pop_size):
                if self.fitness[i] < self.p_best_fitness[i]:
                    self.p_best[i] = self.population[i]
                    self.p_best_fitness[i] = self.fitness[i]
            
            min_index = np.argmin(self.p_best_fitness)
            if self.p_best_fitness[min_index] < self.g_best_fitness:
                self.g_best = self.p_best[min_index]
                self.g_best_fitness = self.p_best_fitness[min_index]
            
            for i in range(self.pop_size):
                indices = np.random.choice(self.pop_size, 3, replace=False)
                a, b, c = self.population[indices]
                F = adapt_rate * np.random.rand()  # Adaptive parameter tuning
                mutant = np.clip(a + F * (b - c), self.lower_bound, self.upper_bound)
                
                cross_points = np.random.rand(self.dim) < 0.9
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, self.population[i])
                
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    adapt_rate = max(0.4, adapt_rate * 0.98)  # Adaptation mechanism

                if evaluations >= self.budget:
                    break
            
            w = 0.5 + np.random.rand() / 2
            c1 = 1.496
            c2 = 1.496
            for i in range(self.pop_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                self.velocities[i] = (w * self.velocities[i] + 
                                      c1 * r1 * (self.p_best[i] - self.population[i]) + 
                                      c2 * r2 * (self.g_best - self.population[i]))
                
                self.population[i] = np.clip(self.population[i] + self.velocities[i], 
                                             self.lower_bound, self.upper_bound)
                self.fitness[i] = np.inf
                
                if evaluations >= self.budget:
                    break
        
        return self.g_best