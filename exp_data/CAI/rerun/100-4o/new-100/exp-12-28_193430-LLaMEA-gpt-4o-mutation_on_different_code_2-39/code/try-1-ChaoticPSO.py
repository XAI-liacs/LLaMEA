import numpy as np

class ChaoticPSO:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.swarm_size = 42  # Changed from 40 to 42
        self.w = 0.5
        self.c1 = 1.5
        self.c2 = 1.5
        self.lb = -5.0
        self.ub = 5.0

    def chaotic_map(self, x):
        return (4 * x * (1 - x)) % 1  # Logistic map for chaos

    def __call__(self, func):
        # Initialize swarm
        x = np.random.uniform(self.lb, self.ub, (self.swarm_size, self.dim))
        v = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        pbest = x.copy()
        pbest_f = np.array([func(xi) for xi in x])
        gbest = pbest[np.argmin(pbest_f)]
        gbest_f = np.min(pbest_f)

        evaluations = self.swarm_size

        while evaluations < self.budget:
            for i in range(self.swarm_size):
                # Update velocity
                r1 = self.chaotic_map(np.random.rand(self.dim))
                r2 = self.chaotic_map(np.random.rand(self.dim))
                v[i] = (self.w * v[i] +
                        self.c1 * r1 * (pbest[i] - x[i]) +
                        self.c2 * r2 * (gbest - x[i]))
                # Update position
                x[i] = x[i] + v[i]
                x[i] = np.clip(x[i], self.lb, self.ub)

                # Evaluate
                f = func(x[i])
                evaluations += 1

                # Update personal best
                if f < pbest_f[i]:
                    pbest[i] = x[i]
                    pbest_f[i] = f

                # Update global best
                if f < gbest_f:
                    gbest = x[i]
                    gbest_f = f
                    self.x_opt = gbest
                    self.f_opt = gbest_f

                if evaluations >= self.budget:
                    break

        return self.f_opt, self.x_opt