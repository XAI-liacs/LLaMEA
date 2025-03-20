import numpy as np

class HybridPSO_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 15
        self.w_max = 0.9
        self.w_min = 0.4
        self.c1 = 1.5
        self.c2 = 1.5
        self.T_init = 1.0
        self.T_min = 1e-3
        self.alpha = 0.95
        self.chaotic_map = np.random.rand(self.num_particles)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        X = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        V = np.random.uniform(-0.5, 0.5, (self.num_particles, self.dim))
        P_best = X.copy()
        P_best_values = np.array([func(x) for x in P_best])
        G_best = P_best[np.argmin(P_best_values)]
        G_best_value = min(P_best_values)

        eval_count = self.num_particles
        T = self.T_init

        while eval_count < self.budget:
            r1, r2 = np.random.rand(self.num_particles, self.dim), np.random.rand(self.num_particles, self.dim)
            w = self.w_max - ((self.w_max - self.w_min) * (eval_count / self.budget))  # Dynamic inertia
            adaptive_velocity = 1 + (np.sin(np.pi * eval_count / self.budget))
            V = w * V + self.c1 * r1 * (P_best - X) + self.c2 * r2 * (G_best - X)
            V *= adaptive_velocity 
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
                new_pos = X[i] + np.random.normal(0, 1, self.dim) * (ub - lb) / 20
                new_pos += np.random.standard_cauchy(self.dim) * (ub - lb) / 100  # Lévy flight
                new_pos = np.clip(new_pos, lb, ub)
                chaotic_factor = self.chaotic_map[i] + (0.1 * np.sin(np.pi * eval_count / self.budget))
                new_value = func(new_pos * chaotic_factor)
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