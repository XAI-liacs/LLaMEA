import numpy as np

class Hybrid_PSO_DE:
    def __init__(self, budget, dim, swarm_size=20, inertia=0.5, cognitive=1.5, social=1.5, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.F = F  # Differential weight
        self.CR = CR  # Crossover probability
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        particles = self.initialize_particles(self.swarm_size, lb, ub)
        velocities = self.initialize_velocities(self.swarm_size)
        personal_best_positions = np.copy(particles)
        personal_best_values = np.full(self.swarm_size, np.inf)
        global_best_position = None
        global_best_value = np.inf

        while self.evaluations < self.budget:
            for i in range(self.swarm_size):
                value = func(particles[i])
                self.evaluations += 1
                
                # Update personal best
                if value < personal_best_values[i]:
                    personal_best_values[i] = value
                    personal_best_positions[i] = np.copy(particles[i])

                # Update global best
                if value < global_best_value:
                    global_best_value = value
                    global_best_position = np.copy(particles[i])

                # PSO update
                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                velocities[i] = (self.inertia * velocities[i] +
                                 self.cognitive * r1 * (personal_best_positions[i] - particles[i]) +
                                 self.social * r2 * (global_best_position - particles[i] if global_best_position is not None else 0))
                particles[i] = np.clip(particles[i] + velocities[i], lb, ub)

                # DE mutation and crossover
                if np.random.rand() < self.CR:
                    indices = np.random.choice(self.swarm_size, 3, replace=False)
                    donor_vector = particles[indices[0]] + self.F * (particles[indices[1]] - particles[indices[2]])
                    mutant_vector = np.where(np.random.rand(self.dim) < self.CR, donor_vector, particles[i])
                    particles[i] = np.clip(mutant_vector, lb, ub)

                if self.evaluations >= self.budget:
                    break

        return global_best_position

    def initialize_particles(self, swarm_size, lb, ub):
        return np.random.uniform(lb, ub, (swarm_size, self.dim))

    def initialize_velocities(self, swarm_size):
        return np.random.uniform(-1, 1, (swarm_size, self.dim))