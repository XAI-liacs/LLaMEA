import numpy as np

class HybridPSODE:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.swarm_size = 50
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.9  # Changed initial inertia weight
        self.final_w = 0.4  # Added final inertia weight for adaptation
        self.F = 0.8
        self.CR = 0.9

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        particles = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(lb-ub, ub-lb, (self.swarm_size, self.dim))
        p_best = particles.copy()
        p_best_values = np.full(self.swarm_size, np.inf)

        for i in range(self.swarm_size):
            f = func(p_best[i])
            p_best_values[i] = f
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = p_best[i]

        g_best = self.x_opt.copy()
        eval_count = self.swarm_size

        while eval_count < self.budget:
            # Adaptive inertia weight calculation
            self.w = self.final_w + (0.9 - self.final_w) * ((self.budget - eval_count) / self.budget)
            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(2, self.dim)
                velocities[i] = (self.w * velocities[i] + 
                                self.c1 * r1 * (p_best[i] - particles[i]) + 
                                self.c2 * r2 * (g_best - particles[i]))

                particles[i] = particles[i] + velocities[i]
                particles[i] = np.clip(particles[i], lb, ub)
                
                f = func(particles[i])
                eval_count += 1

                if f < p_best_values[i]:
                    p_best[i] = particles[i]
                    p_best_values[i] = f

                    if f < self.f_opt:
                        self.f_opt = f
                        self.x_opt = particles[i]

            if eval_count < self.budget:
                for i in range(self.swarm_size):
                    indices = list(range(self.swarm_size))
                    indices.remove(i)
                    a, b, c = np.random.choice(indices, 3, replace=False)
                    mutant = p_best[a] + self.F * (p_best[b] - p_best[c])
                    mutant = np.clip(mutant, lb, ub)

                    trial = np.copy(particles[i])
                    jrand = np.random.randint(self.dim)
                    for j in range(self.dim):
                        if np.random.rand() < self.CR or j == jrand:
                            trial[j] = mutant[j]

                    f = func(trial)
                    eval_count += 1

                    if f < p_best_values[i]:
                        p_best[i] = trial
                        p_best_values[i] = f

                        if f < self.f_opt:
                            self.f_opt = f
                            self.x_opt = trial

                    # Local search mechanism
                    if np.random.rand() < 0.1 and eval_count < self.budget:
                        local_candidate = trial + np.random.normal(0, 0.1, self.dim)
                        local_candidate = np.clip(local_candidate, lb, ub)
                        f_local = func(local_candidate)
                        eval_count += 1
                        if f_local < p_best_values[i]:
                            p_best[i] = local_candidate
                            p_best_values[i] = f_local
                            if f_local < self.f_opt:
                                self.f_opt = f_local
                                self.x_opt = local_candidate

        return self.f_opt, self.x_opt