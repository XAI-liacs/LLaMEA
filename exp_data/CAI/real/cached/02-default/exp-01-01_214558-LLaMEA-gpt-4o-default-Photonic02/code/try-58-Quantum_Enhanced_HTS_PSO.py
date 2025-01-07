import numpy as np
from collections import deque

class Quantum_Enhanced_HTS_PSO:
    def __init__(self, budget, dim, base_group_size=10, inertia=0.5, cognitive=1.5, social=1.5, memory_size=5, quantum_prob=0.2):
        self.budget = budget
        self.dim = dim
        self.base_group_size = base_group_size
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.memory_size = memory_size
        self.quantum_prob = quantum_prob
        self.evaluations = 0
        self.tabu_list = deque(maxlen=self.memory_size)
        self.learning_rate = 0.5
        self.dynamic_grouping = True

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_global_position = None
        best_global_value = float('inf')
        
        group_size = self.base_group_size
        particles = self.initialize_particles(group_size, lb, ub)
        velocities = self.initialize_velocities(group_size)
        particle_best_positions = np.copy(particles)
        particle_best_values = np.full(group_size, float('inf'))

        while self.evaluations < self.budget:
            for i in range(group_size):
                position = particles[i]
                if np.random.rand() < self.quantum_prob:
                    position = self.quantum_tunneling(position, lb, ub)

                self.update_learning_rate(best_global_value)
                self.update_inertia()

                neighborhood_best_position = self.get_neighborhood_best(particles, particle_best_values)

                velocities[i] = (self.inertia * velocities[i] +
                                 self.cognitive * np.random.random(self.dim) * (particle_best_positions[i] - position) +
                                 self.social * np.random.random(self.dim) * (neighborhood_best_position - position))
                position = np.clip(position + self.learning_rate * velocities[i], lb, ub)
                particles[i] = position

                if tuple(position) in self.tabu_list:
                    continue

                value = func(position)
                self.evaluations += 1
                self.tabu_list.append(tuple(position))

                if value < particle_best_values[i]:
                    particle_best_values[i] = value
                    particle_best_positions[i] = position

                if value < best_global_value:
                    best_global_value = value
                    best_global_position = position

                if self.evaluations >= self.budget:
                    break

        return best_global_position

    def initialize_particles(self, group_size, lb, ub):
        return np.random.uniform(lb, ub, (group_size, self.dim))

    def initialize_velocities(self, group_size):
        return np.random.uniform(-1, 1, (group_size, self.dim))

    def quantum_tunneling(self, position, lb, ub):
        q_position = position + (np.random.rand(self.dim) - 0.5) * (ub - lb) * 0.2
        return np.clip(q_position, lb, ub)

    def update_learning_rate(self, best_value):
        self.learning_rate = max(0.1, min(1.0, 0.5 * (1.0 - self.evaluations / self.budget)))

    def update_inertia(self):
        self.inertia = 0.9 - 0.7 * (self.evaluations / self.budget)

    def get_neighborhood_best(self, particles, particle_best_values):
        if self.dynamic_grouping:
            idx = np.random.choice(len(particles))
            return particles[np.argmin(particle_best_values)]
        return np.mean(particles, axis=0)