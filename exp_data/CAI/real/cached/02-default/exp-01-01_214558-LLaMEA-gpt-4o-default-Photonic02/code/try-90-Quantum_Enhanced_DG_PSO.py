import numpy as np
from collections import deque

class Quantum_Enhanced_DG_PSO:
    def __init__(self, budget, dim, base_group_size=10, inertia=0.5, cognitive=1.5, social=1.5, memory_size=5, quantum_prob=0.2, max_group_size=20):
        self.budget = budget
        self.dim = dim
        self.base_group_size = base_group_size
        self.max_group_size = max_group_size
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.memory_size = memory_size
        self.quantum_prob = quantum_prob
        self.evaluations = 0
        self.tabu_list = deque(maxlen=self.memory_size)
        self.learning_rate = 0.5

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_global_position = None
        best_global_value = float('inf')
        
        group_size = self.base_group_size
        particles = self.initialize_particles(group_size, lb, ub)
        velocities = self.initialize_velocities(group_size)

        while self.evaluations < self.budget:
            diversity = self.calculate_diversity(particles)
            group_size = min(self.max_group_size, self.base_group_size + int(diversity * 10))
            particles, velocities = self.adjust_group_size(particles, velocities, group_size, lb, ub)

            for i in range(group_size):
                position = particles[i]
                
                if np.random.rand() < self.quantum_prob:
                    position = self.adaptive_perturbation(position, lb, ub, diversity)

                self.update_learning_rate(diversity)

                velocities[i] = (self.inertia * velocities[i] +
                                 self.cognitive * np.random.random(self.dim) * (particles[i] - position) +
                                 self.social * np.random.random(self.dim) * (best_global_position - position if best_global_position is not None else 0))
                position = np.clip(position + self.learning_rate * velocities[i], lb, ub)
                particles[i] = position

                if tuple(position) in self.tabu_list:
                    continue

                value = func(position)
                self.evaluations += 1
                self.tabu_list.append(tuple(position))

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

    def adaptive_perturbation(self, position, lb, ub, diversity):
        q_position = position + (np.random.rand(self.dim) - 0.5) * (ub - lb) * 0.1 * diversity
        return np.clip(q_position, lb, ub)

    def calculate_diversity(self, particles):
        mean_position = np.mean(particles, axis=0)
        return np.mean([np.linalg.norm(p - mean_position) for p in particles])

    def update_learning_rate(self, diversity):
        self.learning_rate = max(0.1, min(1.0, 0.5 * (1.0 - self.evaluations / self.budget * np.tanh(diversity))))

    def adjust_group_size(self, particles, velocities, group_size, lb, ub):
        current_size = len(particles)
        if group_size > current_size:
            new_particles = np.random.uniform(lb, ub, (group_size - current_size, self.dim))
            new_velocities = np.random.uniform(-1, 1, (group_size - current_size, self.dim))
            particles = np.vstack((particles, new_particles))
            velocities = np.vstack((velocities, new_velocities))
        elif group_size < current_size:
            particles = particles[:group_size]
            velocities = velocities[:group_size]
        return particles, velocities