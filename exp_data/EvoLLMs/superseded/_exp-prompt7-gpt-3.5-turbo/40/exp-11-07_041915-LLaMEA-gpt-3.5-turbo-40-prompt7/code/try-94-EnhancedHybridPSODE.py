import numpy as np

class EnhancedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        pop_size = 30
        swarm = np.random.uniform(-5.0, 5.0, (pop_size, self.dim))
        velocities = np.zeros((pop_size, self.dim))
        p_best = swarm.copy()
        p_best_scores = np.array([func(ind) for ind in p_best])
        g_best = p_best[p_best_scores.argmin()]
        g_best_score = np.min(p_best_scores)

        for _ in range(self.budget // pop_size):
            for i in range(pop_size):
                r1, r2 = np.random.rand(2)
                velocities[i] = 0.729*velocities[i] + 1.496*r1*(p_best[i] - swarm[i]) + 1.496*r2*(g_best - swarm[i])
                swarm[i] = np.clip(swarm[i] + velocities[i], -5.0, 5.0)
                new_score = func(swarm[i])
                if new_score < p_best_scores[i]:
                    p_best[i], p_best_scores[i] = swarm[i].copy(), new_score
                    if new_score < g_best_score:
                        g_best, g_best_score = swarm[i].copy(), new_score

        return g_best