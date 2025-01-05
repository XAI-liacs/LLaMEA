import numpy as np

class QIPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 50
        self.alpha = 0.75  # Coefficient of the quantum potential
        self.best_global = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        particles = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best = np.copy(particles)
        personal_best_fitness = np.array([func(p) for p in particles])
        best_idx = np.argmin(personal_best_fitness)
        self.best_global = personal_best[best_idx]

        evaluations = self.num_particles

        while evaluations < self.budget:
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = r1 * velocities[i] + r2 * (self.best_global - particles[i])

                # Apply Quantum-inspired update
                quantum_potential = np.random.normal(0, self.alpha, self.dim)
                particles[i] += 0.5 * velocities[i] + quantum_potential

                particles[i] = np.clip(particles[i], lb, ub)

                fitness = func(particles[i])
                evaluations += 1

                if fitness < personal_best_fitness[i]:
                    personal_best[i] = particles[i]
                    personal_best_fitness[i] = fitness

                    if fitness < personal_best_fitness[best_idx]:
                        best_idx = i
                        self.best_global = personal_best[best_idx]

        return self.best_global