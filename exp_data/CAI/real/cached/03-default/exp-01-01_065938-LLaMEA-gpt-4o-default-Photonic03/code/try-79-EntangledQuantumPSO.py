import numpy as np

class EntangledQuantumPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 50
        self.alpha = 0.5  # Weight for cognitive component
        self.beta = 0.3  # Weight for social component
        self.gamma = 0.2  # Weight for quantum entanglement
        self.history = []

    def quantum_entanglement(self, particles, best_global):
        entangled_particles = []
        for particle in particles:
            delta = np.random.uniform(-1, 1, self.dim)
            entangled_particle = particle + self.gamma * delta * (best_global - particle)
            entangled_particles.append(entangled_particle)
        return np.array(entangled_particles)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        particles = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-0.1, 0.1, (self.swarm_size, self.dim))
        personal_best = particles.copy()
        personal_best_fitness = np.array([func(x) for x in personal_best])
        best_idx = np.argmin(personal_best_fitness)
        best_global = personal_best[best_idx]

        evaluations = self.swarm_size

        while evaluations < self.budget:
            entangled_particles = self.quantum_entanglement(particles, best_global)
            for i in range(self.swarm_size):
                r1, r2 = np.random.uniform(size=2)
                velocities[i] = (self.alpha * velocities[i]
                                 + self.beta * r1 * (personal_best[i] - particles[i])
                                 + self.beta * r2 * (best_global - particles[i]))

                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], lb, ub)

                fitness = func(particles[i])
                evaluations += 1

                if fitness < personal_best_fitness[i]:
                    personal_best[i] = particles[i]
                    personal_best_fitness[i] = fitness
                    if fitness < personal_best_fitness[best_idx]:
                        best_idx = i
                        best_global = particles[i]

            # Quantum entangled update
            particles = (1 - self.gamma) * particles + self.gamma * entangled_particles

            self.history.append(best_global)

        return best_global