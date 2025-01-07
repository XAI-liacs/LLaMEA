import numpy as np
from collections import deque

class Adaptive_Quantum_Swarm:
    def __init__(self, budget, dim, swarm_size=15, inertia=0.7, cognitive=1.4, social=1.4, memory_size=5, quantum_prob=0.3):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.memory_size = memory_size
        self.quantum_prob = quantum_prob
        self.evaluations = 0
        self.tabu_list = deque(maxlen=self.memory_size)
        self.learning_rate = 0.6

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_global_position = None
        best_global_value = float('inf')
        
        particles = self.initialize_particles(lb, ub)
        velocities = self.initialize_velocities()

        while self.evaluations < self.budget:
            for i in range(self.swarm_size):
                position = particles[i]
                
                if np.random.rand() < self.quantum_prob:
                    position = self.adaptive_quantum_perturbation(position, lb, ub, particles)

                self.update_learning_rate()

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

    def initialize_particles(self, lb, ub):
        return np.random.uniform(lb, ub, (self.swarm_size, self.dim))

    def initialize_velocities(self):
        return np.random.uniform(-1, 1, (self.swarm_size, self.dim))

    def adaptive_quantum_perturbation(self, position, lb, ub, particles):
        neighbor_avg = np.mean(particles, axis=0)
        q_position = position + (np.random.rand(self.dim) - 0.5) * (ub - lb) * 0.05 + (neighbor_avg - position) * 0.05
        return np.clip(q_position, lb, ub)

    def update_learning_rate(self):
        self.learning_rate = max(0.2, min(0.9, 0.6 * (1.0 - self.evaluations / self.budget)))