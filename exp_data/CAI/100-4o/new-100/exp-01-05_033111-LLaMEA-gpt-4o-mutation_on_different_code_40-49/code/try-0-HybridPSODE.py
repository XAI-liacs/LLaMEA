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
        self.w = 0.7
        self.F = 0.8
        self.CR = 0.9

    def __call__(self, func):
        # Initialize particles
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
            for i in range(self.swarm_size):
                # Update velocity
                r1, r2 = np.random.rand(2, self.dim)
                velocities[i] = (self.w * velocities[i] + 
                                self.c1 * r1 * (p_best[i] - particles[i]) + 
                                self.c2 * r2 * (g_best - particles[i]))

                # Update position
                particles[i] = particles[i] + velocities[i]

                # Clamp position within bounds
                particles[i] = np.clip(particles[i], lb, ub)

                # Evaluate new position
                f = func(particles[i])
                eval_count += 1

                # Update personal best
                if f < p_best_values[i]:
                    p_best[i] = particles[i]
                    p_best_values[i] = f

                    # Update global best
                    if f < self.f_opt:
                        self.f_opt = f
                        self.x_opt = particles[i]

            # Differential Evolution Mutation and Crossover
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

                    # Evaluate trial
                    f = func(trial)
                    eval_count += 1

                    # Selection
                    if f < p_best_values[i]:
                        p_best[i] = trial
                        p_best_values[i] = f

                        # Update global best
                        if f < self.f_opt:
                            self.f_opt = f
                            self.x_opt = trial

        return self.f_opt, self.x_opt