import numpy as np

class HybridSwarmEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = int(np.sqrt(self.budget))
        self.pbest = None
        self.gbest = None
        self.velocity = None
        self.elite_fraction = 0.2  # New parameter for elite retention, change 1/17
    
    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        population = np.random.uniform(bounds[0], bounds[1], (self.population_size, self.dim))
        self.velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.pbest = population.copy()
        self.gbest = population[np.argmin([func(ind) for ind in population])]
        
        eval_count = self.population_size
        
        while eval_count < self.budget:
            adaptive_scale = 1 - (eval_count / self.budget)  # Adaptive velocity scaling, change 2/17
            elite_count = int(self.population_size * self.elite_fraction)  # Number of elites, change 3/17
            elite_indices = np.argsort([func(ind) for ind in population])[:elite_count]  # Select elites, change 4/17
            for i in range(self.population_size):
                if i not in elite_indices:
                    r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                    self.velocity[i] = adaptive_scale * (0.5 * self.velocity[i] + 
                                                        r1 * (self.pbest[i] - population[i]) + 
                                                        r2 * (self.gbest - population[i]))  # Apply scaling, change 5/17
                    population[i] += self.velocity[i]
                    population[i] = np.clip(population[i], bounds[0], bounds[1])
                
                    if func(population[i]) < func(self.pbest[i]):
                        self.pbest[i] = population[i]
                
                    if func(population[i]) < func(self.gbest):
                        self.gbest = population[i]
                    
                    eval_count += 1
                    if eval_count >= self.budget:
                        break
        
        return self.gbest