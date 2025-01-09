import numpy as np

class PSO_SA_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n_particles = 10
        max_iter = self.budget // n_particles

        # Initialize particle positions and velocities
        particles_pos = np.random.uniform(-5.0, 5.0, (n_particles, self.dim))
        particles_vel = np.random.uniform(-0.1, 0.1, (n_particles, self.dim))
        personal_best = particles_pos.copy()
        global_best = particles_pos[np.argmin([func(p) for p in particles_pos])]

        for _ in range(max_iter):
            T = 1.0 - _ / max_iter  # Temperature for SA
            for i in range(n_particles):
                # Update particle position using PSO
                particles_vel[i] = 0.5 * particles_vel[i] + 2 * np.random.rand() * (personal_best[i] - particles_pos[i]) + \
                                    2 * np.random.rand() * (global_best - particles_pos[i])
                particles_pos[i] = np.clip(particles_pos[i] + particles_vel[i], -5.0, 5.0)

                # Perform Simulated Annealing
                candidate = particles_pos[i] + np.random.normal(0, 0.1, self.dim)
                if func(candidate) < func(particles_pos[i]) or np.random.rand() < np.exp((func(particles_pos[i]) - func(candidate)) / T):
                    particles_pos[i] = candidate

                # Update personal and global best
                if func(particles_pos[i]) < func(personal_best[i]):
                    personal_best[i] = particles_pos[i]
                if func(particles_pos[i]) < func(global_best):
                    global_best = particles_pos[i]

        return global_best