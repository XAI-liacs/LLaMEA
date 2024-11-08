import numpy as np

class EnhancedPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 30
        self.inertia_min = 0.4
        self.inertia_max = 0.9
        self.c1 = 2.0
        self.c2 = 2.0
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        swarm = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, self.dim))
        velocities = np.zeros((self.swarm_size, self.dim))
        pbest_positions = swarm.copy()
        pbest_scores = np.full(self.swarm_size, np.inf)
        gbest_position = np.zeros(self.dim)
        gbest_score = np.inf
        inertia_weight = self.inertia_max

        for _ in range(self.budget):
            scores = np.array([func(p) for p in swarm])
            improve_pbest = scores < pbest_scores
            pbest_scores[improve_pbest] = scores[improve_pbest]
            pbest_positions[improve_pbest] = swarm[improve_pbest].copy()
            best_index = np.argmin(scores)
            if scores[best_index] < gbest_score:
                gbest_score = scores[best_index]
                gbest_position = swarm[best_index].copy()

            r1_r2 = np.random.uniform(0, 1, (self.swarm_size, self.dim, 2))
            velocities = inertia_weight * velocities + r1_r2[:,:,0] * self.c1 * (pbest_positions - swarm) + r1_r2[:,:,1] * self.c2 * (gbest_position - swarm)
            swarm += velocities

            swarm = np.clip(swarm, self.lower_bound, self.upper_bound)

            inertia_weight = self.inertia_max - (_ / self.budget) * (self.inertia_max - self.inertia_min)
        
        return gbest_position