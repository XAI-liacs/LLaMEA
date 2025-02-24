import numpy as np

class QuantumInspiredParticleSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        num_particles = 10 * self.dim
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        swarm = np.random.uniform(bounds[0], bounds[1], (num_particles, self.dim))
        velocities = np.zeros((num_particles, self.dim))
        personal_best = np.copy(swarm)
        personal_best_fitness = np.array([func(ind) for ind in personal_best])
        global_best_idx = np.argmin(personal_best_fitness)
        global_best = personal_best[global_best_idx]
        global_best_fitness = personal_best_fitness[global_best_idx]

        omega = 0.5  # Inertia weight
        phi_p = 1.5  # Personal attraction coefficient
        phi_g = 1.5  # Global attraction coefficient
        q_tunneling_intensity = 0.1  # Intensity of quantum tunneling

        for iteration in range(self.budget - num_particles):
            r_p = np.random.rand(num_particles, self.dim)
            r_g = np.random.rand(num_particles, self.dim)
            velocities = (omega * velocities +
                          phi_p * r_p * (personal_best - swarm) +
                          phi_g * r_g * (global_best - swarm))

            for i in range(num_particles):
                if np.random.rand() < q_tunneling_intensity:
                    # Quantum Tunneling - Jump to a random position
                    swarm[i] = np.random.uniform(bounds[0], bounds[1], self.dim)
                else:
                    # Normal velocity update
                    swarm[i] = np.clip(swarm[i] + velocities[i], bounds[0], bounds[1])
                
                fitness_i = func(swarm[i])

                if fitness_i < personal_best_fitness[i]:
                    personal_best[i] = swarm[i]
                    personal_best_fitness[i] = fitness_i
                    if fitness_i < global_best_fitness:
                        global_best = swarm[i]
                        global_best_fitness = fitness_i

        return global_best, global_best_fitness