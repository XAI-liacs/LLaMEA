import numpy as np
from collections import deque

class Hierarchical_Quantum_Enhanced_HTS_PSO:
    def __init__(self, budget, dim, base_group_size=10, inertia=0.5, cognitive=1.5, social=1.5, memory_size=5, quantum_prob=0.2, mutation_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.base_group_size = base_group_size
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.memory_size = memory_size
        self.quantum_prob = quantum_prob
        self.mutation_rate = mutation_rate
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
        personal_best_positions = np.copy(particles)
        personal_best_values = np.full(group_size, float('inf'))

        while self.evaluations < self.budget:
            for i in range(group_size):
                position = particles[i]
                
                if np.random.rand() < self.quantum_prob:
                    position = self.quantum_perturbation(position, lb, ub)

                self.update_learning_rate()

                velocities[i] = (self.inertia * velocities[i] +
                                 self.cognitive * np.random.random(self.dim) * (personal_best_positions[i] - position) +
                                 self.social * np.random.random(self.dim) * (best_global_position - position if best_global_position is not None else 0))
                position = np.clip(position + self.learning_rate * velocities[i], lb, ub)
                particles[i] = position

                if tuple(position) in self.tabu_list:
                    continue

                value = func(position)
                self.evaluations += 1
                self.tabu_list.append(tuple(position))
                
                if value < personal_best_values[i]:
                    personal_best_values[i] = value
                    personal_best_positions[i] = position

                if value < best_global_value:
                    best_global_value = value
                    best_global_position = position

                if self.evaluations >= self.budget:
                    break

                # Apply adaptive mutation based on diversity
                if np.random.rand() < self.mutation_rate:
                    particles[i] = self.adaptive_mutation(particles[i], lb, ub)

        return best_global_position

    def initialize_particles(self, group_size, lb, ub):
        return np.random.uniform(lb, ub, (group_size, self.dim))

    def initialize_velocities(self, group_size):
        return np.random.uniform(-1, 1, (group_size, self.dim))

    def quantum_perturbation(self, position, lb, ub):
        q_position = position + (np.random.rand(self.dim) - 0.5) * (ub - lb) * 0.1
        return np.clip(q_position, lb, ub)

    def update_learning_rate(self):
        diversity = np.std([np.linalg.norm(p) for p in self.tabu_list]) if len(self.tabu_list) > 0 else 0
        self.learning_rate = max(0.1, min(1.0, 0.5 * (1.0 - self.evaluations / self.budget * np.tanh(diversity))))

    def adaptive_mutation(self, position, lb, ub):
        mutation_strength = (1 - (self.evaluations / self.budget)) * 0.1
        mutated_position = position + np.random.normal(0, mutation_strength, size=self.dim)
        return np.clip(mutated_position, lb, ub)