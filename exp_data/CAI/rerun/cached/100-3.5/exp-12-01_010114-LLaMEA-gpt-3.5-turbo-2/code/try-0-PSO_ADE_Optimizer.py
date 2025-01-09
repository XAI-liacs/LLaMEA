import numpy as np

class PSO_ADE_Optimizer:
    def __init__(self, budget, dim, population_size=30, max_iter=1000, c1=2.05, c2=2.05, f=0.5, k=0.5):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.max_iter = max_iter
        self.c1 = c1
        self.c2 = c2
        self.f = f
        self.k = k
        
    def __call__(self, func):
        def pso_ade(func, budget, dim, population_size, max_iter, c1, c2, f, k):
            # PSO Initialization
            swarm_position = np.random.uniform(-5.0, 5.0, (population_size, dim))
            swarm_velocity = np.zeros((population_size, dim))
            pbest_position = swarm_position.copy()
            pbest_value = np.full(population_size, np.inf)
            gbest_position = np.zeros(dim)
            gbest_value = np.inf
            
            # DE Initialization
            candidates = np.random.uniform(-5.0, 5.0, (3*population_size, dim))
            candidates_values = np.array([func(candidate) for candidate in candidates])
            for i in range(max_iter):
                for j in range(population_size):
                    # PSO Update
                    swarm_velocity[j] = f * swarm_velocity[j] + c1 * np.random.rand() * (pbest_position[j] - swarm_position[j]) + c2 * np.random.rand() * (gbest_position - swarm_position[j])
                    swarm_position[j] = np.clip(swarm_position[j] + swarm_velocity[j], -5.0, 5.0)
                    
                    # ADE Update
                    indices = np.random.choice(3*population_size, 3, replace=False)
                    candidate = candidates[indices[0]] + k * (candidates[indices[1]] - candidates[indices[2]])
                    candidate_value = func(candidate)
                    if candidate_value < candidates_values[indices[0]]:
                        candidates[indices[0]] = candidate
                        candidates_values[indices[0]] = candidate_value
                        
                    # Update pbest and gbest
                    if candidates_values[indices[0]] < pbest_value[j]:
                        pbest_value[j] = candidates_values[indices[0]]
                        pbest_position[j] = candidates[indices[0]]
                    if pbest_value[j] < gbest_value:
                        gbest_value = pbest_value[j]
                        gbest_position = pbest_position[j]
                        
                    budget -= 1
                    if budget == 0:
                        return gbest_value
                    
            return gbest_value
        
        return pso_ade(func, self.budget, self.dim, self.population_size, self.max_iter, self.c1, self.c2, self.f, self.k)