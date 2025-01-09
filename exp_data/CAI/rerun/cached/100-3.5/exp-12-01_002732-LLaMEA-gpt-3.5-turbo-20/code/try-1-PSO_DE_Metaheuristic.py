import numpy as np

class PSO_DE_Metaheuristic:
    def __init__(self, budget, dim, swarm_size=20, pso_iters=100, de_iters=100, pso_w=0.5, pso_c1=1.5, pso_c2=1.5, de_cr=0.9, de_f=0.8):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.pso_iters = pso_iters
        self.de_iters = de_iters
        self.pso_w = pso_w
        self.pso_c1 = pso_c1
        self.pso_c2 = pso_c2
        self.de_cr = de_cr
        self.de_f = de_f

    def __call__(self, func):
        def fitness(x):
            return func(x)
        
        def pso_update_position(x, v):
            return x + v
        
        def de_mutation(x, v1, v2, v3):
            return x + self.de_f * (v1 - x) + self.de_f * (v2 - v3)
        
        def bound_check(x):
            return np.clip(x, -5.0, 5.0)
        
        # Initialization
        swarm = np.random.uniform(-5.0, 5.0, size=(self.swarm_size, self.dim))
        velocity = np.zeros((self.swarm_size, self.dim))
        pbest = swarm.copy()
        pbest_fitness = np.array([fitness(p) for p in pbest])
        gbest_idx = np.argmin(pbest_fitness)
        gbest = pbest[gbest_idx]
        
        # PSO optimization
        for _ in range(self.pso_iters):
            r1, r2 = np.random.rand(self.swarm_size, self.dim), np.random.rand(self.swarm_size, self.dim)
            velocity = self.pso_w * velocity + self.pso_c1 * r1 * (pbest - swarm) + self.pso_c2 * r2 * (gbest - swarm)
            swarm = pso_update_position(swarm, velocity)
            swarm = np.array([bound_check(p) for p in swarm])
            current_fitness = np.array([fitness(p) for p in swarm])
            update_indices = current_fitness < pbest_fitness
            pbest[update_indices] = swarm[update_indices]
            pbest_fitness[update_indices] = current_fitness[update_indices]
            gbest_idx = np.argmin(pbest_fitness)
            gbest = pbest[gbest_idx]
        
        # DE optimization
        for _ in range(self.de_iters):
            for i in range(self.swarm_size):
                candidates = np.delete(swarm, i, axis=0)
                idxs = np.random.choice(len(candidates), 3, replace=False)
                a, b, c = candidates[idxs]
                new_point = de_mutation(swarm[i], a, b, c)
                new_point = bound_check(new_point)
                if fitness(new_point) < fitness(swarm[i]):
                    swarm[i] = new_point
            current_fitness = np.array([fitness(p) for p in swarm])
            update_indices = current_fitness < pbest_fitness
            pbest[update_indices] = swarm[update_indices]
            pbest_fitness[update_indices] = current_fitness[update_indices]
            gbest_idx = np.argmin(pbest_fitness)
            gbest = pbest[gbest_idx]
        
        return gbest