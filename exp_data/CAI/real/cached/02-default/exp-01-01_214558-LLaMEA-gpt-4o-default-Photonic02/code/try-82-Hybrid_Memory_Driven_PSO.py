import numpy as np
from collections import deque

class Hybrid_Memory_Driven_PSO:
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
        self.memory_archive = []
        self.memory_threshold = 1e-5

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_global_position = None
        best_global_value = float('inf')
        
        group_size = self.base_group_size
        particles = self.initialize_particles(group_size, lb, ub)
        velocities = self.initialize_velocities(group_size)

        while self.evaluations < self.budget:
            for i in range(group_size):
                position = particles[i]
                
                if np.random.rand() < self.quantum_prob:
                    position = self.quantum_perturbation(position, lb, ub)

                self.update_learning_rate(best_global_value)
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
                self.update_memory_archive(position, value)

                if value < best_global_value:
                    best_global_value = value
                    best_global_position = position

                if self.evaluations >= self.budget:
                    break

            self.dynamic_group_adjustment()

        return best_global_position

    def initialize_particles(self, group_size, lb, ub):
        return np.random.uniform(lb, ub, (group_size, self.dim))

    def initialize_velocities(self, group_size):
        return np.random.uniform(-1, 1, (group_size, self.dim))

    def quantum_perturbation(self, position, lb, ub):
        q_position = position + (np.random.rand(self.dim) - 0.5) * (ub - lb) * 0.1
        return np.clip(q_position, lb, ub)

    def update_learning_rate(self, best_value):
        self.learning_rate = max(0.1, min(1.0, 0.5 * (1.0 - self.evaluations / self.budget)))

    def update_memory_archive(self, position, value):
        # Add to memory archive if significant improvement
        if not self.memory_archive or (self.memory_archive and value < min(v for _, v in self.memory_archive) - self.memory_threshold):
            self.memory_archive.append((position, value))
            self.memory_archive = sorted(self.memory_archive, key=lambda x: x[1])[:self.memory_size]

    def dynamic_group_adjustment(self):
        # Adjust group size based on diversity in memory archive
        diversity = np.std([v for _, v in self.memory_archive]) if self.memory_archive else 0
        if diversity < self.memory_threshold and len(self.memory_archive) >= self.memory_size:
            self.base_group_size = max(5, self.base_group_size - 1)
        else:
            self.base_group_size = min(20, self.base_group_size + 1)