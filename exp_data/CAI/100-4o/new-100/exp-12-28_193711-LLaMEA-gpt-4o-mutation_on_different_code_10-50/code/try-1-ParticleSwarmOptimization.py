import numpy as np

class ParticleSwarmOptimization:
    def __init__(self, budget=10000, dim=10, swarm_size=30, c1=2.0, c2=2.0, w=0.9):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.c1 = c1  # cognitive coefficient
        self.c2 = c2  # social coefficient
        self.w = w    # initial inertia weight
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        swarm = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(swarm)
        personal_best_scores = np.full(self.swarm_size, np.inf)
        global_best_position = None
        global_best_score = np.inf

        for _ in range(self.budget // self.swarm_size):
            for i in range(self.swarm_size):
                f = func(swarm[i])
                if f < personal_best_scores[i]:
                    personal_best_scores[i] = f
                    personal_best_positions[i] = swarm[i]
                if f < global_best_score:
                    global_best_score = f
                    global_best_position = swarm[i]

            # Adaptive inertia weight adjustment
            self.w = 0.9 - 0.5 * (_ / (self.budget // self.swarm_size))

            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - swarm[i]) +
                                 self.c2 * r2 * (global_best_position - swarm[i]))
                swarm[i] = np.clip(swarm[i] + velocities[i], lb, ub)

        self.f_opt = global_best_score
        self.x_opt = global_best_position
        return self.f_opt, self.x_opt