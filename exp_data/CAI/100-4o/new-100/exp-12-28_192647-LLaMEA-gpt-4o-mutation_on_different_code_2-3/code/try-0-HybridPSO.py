import numpy as np

class HybridPSO:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.num_particles = 50
        self.c1 = 1.5  # cognitive coefficient
        self.c2 = 1.5  # social coefficient
        self.w = 0.5   # inertia weight
        self.diff_weight = 0.8  # weight for differential mutation
        self.cr = 0.9   # crossover probability

    def __call__(self, func):
        # Initialize particle positions and velocities
        lb, ub = func.bounds.lb, func.bounds.ub
        particles = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.full(self.num_particles, np.Inf)

        # Evaluate initial population
        for i in range(self.num_particles):
            f = func(particles[i])
            if f < personal_best_scores[i]:
                personal_best_scores[i] = f
                personal_best_positions[i] = particles[i]
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = particles[i]
        eval_count = self.num_particles

        # Main optimization loop
        while eval_count < self.budget:
            # Update particle velocities and positions
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - particles[i]) +
                                 self.c2 * r2 * (self.x_opt - particles[i]))
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], lb, ub)

            # Apply differential mutation within particle swarm
            for i in range(self.num_particles):
                if eval_count >= self.budget: break
                idxs = [idx for idx in range(self.num_particles) if idx != i]
                a, b, c = particles[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.diff_weight * (b - c), lb, ub)
                trial = np.where(np.random.rand(self.dim) < self.cr, mutant, particles[i])
                f = func(trial)
                eval_count += 1
                
                if f < personal_best_scores[i]:
                    personal_best_scores[i] = f
                    personal_best_positions[i] = trial
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = trial
        
        return self.f_opt, self.x_opt