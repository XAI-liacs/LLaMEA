import numpy as np

class ParticleSwarmOptimization:
    def __init__(self, budget=10000, dim=10, num_particles=30, w=0.5, c1=1.5, c2=1.5):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.w = w  # inertia coefficient
        self.c1 = c1  # cognitive (particle) coefficient
        self.c2 = c2  # social (swarm) coefficient
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        particles = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.full(self.num_particles, np.Inf)
        
        global_best_position = None
        global_best_score = np.Inf

        eval_count = 0
        while eval_count < self.budget:
            for i, particle in enumerate(particles):
                if eval_count >= self.budget:
                    break
                f = func(particle)
                eval_count += 1
                if f < personal_best_scores[i]:
                    personal_best_scores[i] = f
                    personal_best_positions[i] = particle
                    if f < global_best_score:
                        global_best_score = f
                        global_best_position = particle

            for i in range(self.num_particles):
                r1, r2 = np.random.rand(2)
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - particles[i]) +
                                 self.c2 * r2 * (global_best_position - particles[i]))
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], lb, ub)

        return global_best_score, global_best_position