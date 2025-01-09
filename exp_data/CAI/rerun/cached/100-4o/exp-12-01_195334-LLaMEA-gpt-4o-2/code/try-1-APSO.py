import numpy as np

class APSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.swarm_size = 30
        self.c1 = 1.5
        self.c2 = 1.5
        self.w_max = 0.9
        self.w_min = 0.4
        self.vel_max = (self.ub - self.lb) * 0.1
        self.vel_min = -(self.ub - self.lb) * 0.1

    def __call__(self, func):
        # Initialize particles
        swarm_pos = np.random.uniform(self.lb, self.ub, (self.swarm_size, self.dim))
        swarm_vel = np.random.uniform(self.vel_min, self.vel_max, (self.swarm_size, self.dim))
        p_best_pos = np.copy(swarm_pos)
        p_best_val = np.full(self.swarm_size, np.inf)
        g_best_pos = None
        g_best_val = np.inf

        evals = 0

        while evals < self.budget:
            # Evaluate swarm
            for i in range(self.swarm_size):
                if evals >= self.budget:
                    break
                val = func(swarm_pos[i])
                evals += 1

                # Update personal best
                if val < p_best_val[i]:
                    p_best_val[i] = val
                    p_best_pos[i] = swarm_pos[i]

                # Update global best
                if val < g_best_val:
                    g_best_val = val
                    g_best_pos = swarm_pos[i]

            # Adaptive inertia weight
            w = self.w_max - (self.w_max - self.w_min) * evals / self.budget

            # Update velocities and positions
            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(2)
                swarm_vel[i] = (w * swarm_vel[i] +
                                self.c1 * r1 * (p_best_pos[i] - swarm_pos[i]) +
                                self.c2 * r2 * (g_best_pos - swarm_pos[i]))

                # Apply velocity constraints
                swarm_vel[i] = np.clip(swarm_vel[i], self.vel_min, self.vel_max)

                # Update position
                swarm_pos[i] += swarm_vel[i]

                # Apply position constraints
                swarm_pos[i] = np.clip(swarm_pos[i], self.lb, self.ub)

                # Velocity re-initialization to prevent stagnation
                if np.random.rand() < 0.1:  # 10% chance to reinitialize
                    swarm_vel[i] = np.random.uniform(self.vel_min, self.vel_max, self.dim)

        return g_best_pos, g_best_val