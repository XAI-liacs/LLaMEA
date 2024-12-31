import numpy as np

class ParticleSwarmOptimization:
    def __init__(self, budget=10000, dim=10, swarm_size=30, inertia=0.5, cognitive=1.5, social=1.5):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        # Initialize the swarm
        lb, ub = func.bounds.lb, func.bounds.ub
        swarm_positions = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        swarm_velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(swarm_positions)
        personal_best_scores = np.full(self.swarm_size, np.inf)

        for _ in range(self.budget // self.swarm_size):
            for i in range(self.swarm_size):
                # Evaluate fitness
                fitness = func(swarm_positions[i])
                
                # Update personal best
                if fitness < personal_best_scores[i]:
                    personal_best_scores[i] = fitness
                    personal_best_positions[i] = swarm_positions[i]
                
                # Update global best
                if fitness < self.f_opt:
                    self.f_opt = fitness
                    self.x_opt = swarm_positions[i]

            # Update velocities and positions
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            for i in range(self.swarm_size):
                swarm_velocities[i] = (self.inertia * swarm_velocities[i] +
                                       self.cognitive * r1 * (personal_best_positions[i] - swarm_positions[i]) +
                                       self.social * r2 * (self.x_opt - swarm_positions[i]))
                swarm_positions[i] += swarm_velocities[i]
                swarm_positions[i] = np.clip(swarm_positions[i], lb, ub)

        return self.f_opt, self.x_opt