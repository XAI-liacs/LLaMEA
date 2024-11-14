import numpy as np

class DynamicPopulationSizePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.w_min = 0.4
        self.w_max = 0.9
        self.c1 = 1.49445
        self.c2 = 1.49445
        self.pop_size = 20
        self.max_velocity = 0.1 * (5.0 - (-5.0))
    
    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        velocity = np.zeros((self.pop_size, self.dim))
        pbest = population.copy()
        pbest_fitness = np.array([func(ind) for ind in population])
        gbest_idx = np.argmin(pbest_fitness)
        gbest = pbest[gbest_idx]
        
        for _ in range(self.budget):
            w = self.w_max - (_ / self.budget) * (self.w_max - self.w_min)
            r1, r2 = np.random.rand(), np.random.rand()
            
            velocity = w * velocity + self.c1 * r1 * (pbest - population) + self.c2 * r2 * (np.tile(gbest, (self.pop_size, 1)) - population)
            velocity = np.clip(velocity, -self.max_velocity, self.max_velocity)
            population += velocity
            
            fitness = np.array([func(ind) for ind in population])
            update_indices = fitness < pbest_fitness
            pbest[update_indices] = population[update_indices]
            pbest_fitness[update_indices] = fitness[update_indices]
            
            gbest_idx = np.argmin(pbest_fitness)
            gbest = pbest[gbest_idx]
            
            # Dynamic population size adaptation
            if np.std(population) > 0.5:
                self.pop_size = int(1.2 * self.pop_size) if np.random.rand() < 0.5 else int(0.8 * self.pop_size)
                population = np.vstack((population, np.random.uniform(-5.0, 5.0, (self.pop_size - len(population), self.dim))))
                velocity = np.vstack((velocity, np.zeros((self.pop_size - len(velocity), self.dim))))
                pbest = np.vstack((pbest, np.random.uniform(-5.0, 5.0, (self.pop_size - len(pbest), self.dim))))
                pbest_fitness = np.hstack((pbest_fitness, np.array([func(ind) for ind in population[len(pbest):]]))
        
        return gbest