import numpy as np

class AdaptivePSO:
    def __init__(self, budget=10000, dim=10, swarm_size=30):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        swarm = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        velocities = np.zeros((self.swarm_size, self.dim))
        p_best = swarm.copy()
        p_best_values = np.full(self.swarm_size, np.Inf)
        
        g_best = None
        g_best_value = np.Inf

        for eval_count in range(self.budget):
            if eval_count == self.budget: break

            # Evaluate swarm
            for i, particle in enumerate(swarm):
                value = func(particle)
                if value < p_best_values[i]:
                    p_best_values[i] = value
                    p_best[i] = particle
                    if value < g_best_value:
                        g_best_value = value
                        g_best = particle

            # Update velocities and positions
            inertia_weight = 0.9 - (eval_count / self.budget) * 0.5
            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive = 2 * r1 * (p_best[i] - swarm[i])
                social = 2 * r2 * (g_best - swarm[i])
                velocities[i] = inertia_weight * velocities[i] + cognitive + social
                swarm[i] = np.clip(swarm[i] + velocities[i], lb, ub)

            # Store the best solution found
            if g_best_value < self.f_opt:
                self.f_opt = g_best_value
                self.x_opt = g_best

        return self.f_opt, self.x_opt