import numpy as np
from collections import deque

class Quantum_Enhanced_HTS_PSO_Advanced:
    def __init__(self, budget, dim, base_group_size=10, inertia=0.5, cognitive=1.5, social=1.5, memory_size=5, quantum_prob=0.2, mutation_factor=0.8, min_group_size=5):
        self.budget = budget
        self.dim = dim
        self.base_group_size = base_group_size
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.memory_size = memory_size
        self.quantum_prob = quantum_prob
        self.mutation_factor = mutation_factor
        self.min_group_size = min_group_size
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
            self.adjust_group_size()
            for i in range(group_size):
                position = particles[i]
                
                if np.random.rand() < self.quantum_prob:
                    position = self.quantum_perturbation(position, lb, ub)

                self.update_learning_rate()

                velocities[i] = (self.inertia * velocities[i] +
                                 self.cognitive * np.random.random(self.dim) * (particles[i] - position) +
                                 self.social * np.random.random(self.dim) * (best_global_position - position if best_global_position is not None else 0))
                position = np.clip(position + self.learning_rate * velocities[i], lb, ub)
                particles[i] = self.differential_mutation(position, lb, ub, particles, best_global_position)

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

    def quantum_perturbation(self, position, lb, ub):
        q_position = position + (np.random.rand(self.dim) - 0.5) * (ub - lb) * 0.1
        return np.clip(q_position, lb, ub)

    def update_learning_rate(self):
        diversity = np.std([np.linalg.norm(p) for p in self.tabu_list])
        self.learning_rate = max(0.1, min(1.0, 0.5 * (1.0 - self.evaluations / self.budget * np.tanh(diversity))))

    def adjust_group_size(self):
        if self.evaluations < self.budget / 2:
            self.base_group_size = max(self.min_group_size, int(self.base_group_size * (1 + 0.1 * np.sin(2 * np.pi * self.evaluations / self.budget))))
        else:
            self.base_group_size = max(self.min_group_size, int(self.base_group_size * (1 - 0.05)))

    def differential_mutation(self, position, lb, ub, particles, best_global_position):
        idxs = np.random.choice(len(particles), 3, replace=False)
        a, b, c = particles[idxs]
        mutant = a + self.mutation_factor * (b - c)
        trial = np.clip(mutant + self.mutation_factor * (best_global_position - position), lb, ub)
        return np.where(np.random.rand(self.dim) < 0.5, trial, position)