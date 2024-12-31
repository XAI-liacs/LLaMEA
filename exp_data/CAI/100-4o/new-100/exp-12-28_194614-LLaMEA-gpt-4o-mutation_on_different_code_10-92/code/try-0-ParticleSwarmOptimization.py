import numpy as np

class ParticleSwarmOptimization:
    def __init__(self, budget=10000, dim=10, swarm_size=30, c1=2.0, c2=2.0, w=0.7):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.c1 = c1  # Cognitive coefficient
        self.c2 = c2  # Social coefficient
        self.w = w    # Inertia weight
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        particles = np.random.uniform(bounds[0], bounds[1], (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.full(self.swarm_size, np.Inf)

        for _ in range(self.budget // self.swarm_size):
            for i in range(self.swarm_size):
                f = func(particles[i])
                if f < personal_best_scores[i]:
                    personal_best_scores[i] = f
                    personal_best_positions[i] = particles[i]

                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = particles[i]

            global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
            
            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - particles[i]) +
                                 self.c2 * r2 * (global_best_position - particles[i]))
                particles[i] = np.clip(particles[i] + velocities[i], bounds[0], bounds[1])
        
        return self.f_opt, self.x_opt