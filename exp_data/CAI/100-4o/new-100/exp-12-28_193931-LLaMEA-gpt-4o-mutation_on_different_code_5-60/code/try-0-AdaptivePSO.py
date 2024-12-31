import numpy as np

class AdaptivePSO:
    def __init__(self, budget=10000, dim=10, num_particles=30, c1=2.05, c2=2.05):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.c1 = c1
        self.c2 = c2
        self.w_max = 0.9
        self.w_min = 0.4
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        particles = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best = particles.copy()
        personal_best_scores = np.full(self.num_particles, np.Inf)

        global_best = None
        global_best_score = np.Inf

        evals = 0
        
        while evals < self.budget:
            for i in range(self.num_particles):
                f = func(particles[i])
                evals += 1

                if f < personal_best_scores[i]:
                    personal_best_scores[i] = f
                    personal_best[i] = particles[i].copy()

                if f < global_best_score:
                    global_best_score = f
                    global_best = particles[i].copy()

                if evals >= self.budget:
                    break

            if global_best is not None:
                self.f_opt = global_best_score
                self.x_opt = global_best

            w = self.w_max - (self.w_max - self.w_min) * (evals / self.budget)

            for i in range(self.num_particles):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                cognitive_velocity = self.c1 * r1 * (personal_best[i] - particles[i])
                social_velocity = self.c2 * r2 * (global_best - particles[i])
                velocities[i] = w * velocities[i] + cognitive_velocity + social_velocity
                particles[i] = particles[i] + velocities[i]

                particles[i] = np.clip(particles[i], func.bounds.lb, func.bounds.ub)

        return self.f_opt, self.x_opt