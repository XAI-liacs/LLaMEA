import numpy as np
from collections import deque

class Adaptive_Quantum_PSOTabu:
    def __init__(self, budget, dim, base_group_size=10, inertia=0.5, cognitive=1.5, social=1.5, initial_memory_size=5, quantum_prob=0.3):
        self.budget = budget
        self.dim = dim
        self.base_group_size = base_group_size
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.memory_size = initial_memory_size
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

                if value < best_global_value:
                    best_global_value = value
                    best_global_position = position
                    # Dynamically adjust tabu memory size based on convergence
                    self.memory_size = max(2, min(10, int(self.memory_size * 1.1)))
                    self.tabu_list = deque(self.tabu_list, maxlen=self.memory_size)

                if self.evaluations >= self.budget:
                    break

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