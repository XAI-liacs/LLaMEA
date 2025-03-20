import numpy as np

class HybridPSO_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = min(10, dim * 2)  # Adaptive swarm size
        self.w_max = 0.9  # Max inertia weight
        self.w_min = 0.4  # Min inertia weight
        self.c1 = 1.5  # Cognitive coefficient
        self.c2 = 1.5  # Social coefficient
        self.T_init = 1.0  # Initial temperature for SA
        self.T_min = 1e-4  # Minimum temperature for SA
        self.alpha = 0.95  # Cooling rate for SA

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        X = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        V = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        P_best = X.copy()
        P_best_values = np.array([func(x) for x in P_best])
        G_best = P_best[np.argmin(P_best_values)]
        G_best_value = min(P_best_values)
        
        eval_count = self.num_particles
        T = self.T_init

        while eval_count < self.budget:
            w = self.w_max - (self.w_max - self.w_min) * eval_count / self.budget
            r1 = np.random.rand(self.num_particles, self.dim)
            r2 = np.random.rand(self.num_particles, self.dim)
            # Adaptive cognitive and social coefficients
            c1_t = self.c1 * (T / self.T_init)
            c2_t = self.c2 * (1 - T / self.T_init)
            V = w * V + c1_t * r1 * (P_best - X) + c2_t * r2 * (G_best - X)
            X = np.clip(X + V, lb, ub)

            current_values = np.array([func(x) for x in X])
            eval_count += self.num_particles

            for i in range(self.num_particles):
                if current_values[i] < P_best_values[i]:
                    P_best[i] = X[i]
                    P_best_values[i] = current_values[i]
                    if P_best_values[i] < G_best_value:
                        G_best = P_best[i]
                        G_best_value = P_best_values[i]

            for i in range(self.num_particles):
                new_pos = X[i] + np.random.normal(0, 1, self.dim) * (ub - lb) / 10
                new_pos = np.clip(new_pos, lb, ub)
                new_value = func(new_pos)
                eval_count += 1
                if new_value < current_values[i] or np.random.rand() < np.exp((current_values[i] - new_value) / T):
                    X[i] = new_pos
                    current_values[i] = new_value
                    if new_value < P_best_values[i]:
                        P_best[i] = new_pos
                        P_best_values[i] = new_value
                        if new_value < G_best_value:
                            G_best = new_pos
                            G_best_value = new_value

            T = max(self.T_min, T * self.alpha)

        return G_best