import numpy as np

class ParticleSwarmOptimization:
    def __init__(self, budget=10000, dim=10, num_particles=30, w=0.9, c1=1.5, c2=1.5):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.w = w  # inertia weight
        self.c1 = c1  # cognitive component
        self.c2 = c2  # social component
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        # Initialize particle positions and velocities
        positions = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.num_particles, np.inf)

        for iter_num in range(self.budget // self.num_particles):
            for i in range(self.num_particles):
                # Evaluate the function at the current position
                f = func(positions[i])
                if f < personal_best_scores[i]:
                    personal_best_scores[i] = f
                    personal_best_positions[i] = positions[i]
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = positions[i]

            # Update particle velocities and positions
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            for i in range(self.num_particles):
                inertia = self.w * velocities[i]
                cognitive = self.c1 * r1 * (personal_best_positions[i] - positions[i])
                social = self.c2 * r2 * (self.x_opt - positions[i])
                velocities[i] = inertia + cognitive + social
                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(positions[i], func.bounds.lb, func.bounds.ub)

            # Adaptively decrease the inertia weight using a non-linear schedule
            self.w = 0.4 + 0.5 * np.cos(np.pi * iter_num / (self.budget // self.num_particles))

            # Dynamically adjust the number of particles
            if iter_num % 10 == 0 and self.num_particles > 5:
                self.num_particles = max(5, self.num_particles - 1)
                positions = positions[:self.num_particles]
                velocities = velocities[:self.num_particles]
                personal_best_positions = personal_best_positions[:self.num_particles]
                personal_best_scores = personal_best_scores[:self.num_particles]

        return self.f_opt, self.x_opt