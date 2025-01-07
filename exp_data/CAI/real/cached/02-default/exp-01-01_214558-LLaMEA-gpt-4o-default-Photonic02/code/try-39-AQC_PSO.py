import numpy as np

class AQC_PSO:
    def __init__(self, budget, dim, num_particles=30, inertia=0.5, cognitive=1.5, social=1.5, quantum_prob=0.1, crossover_prob=0.3):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.quantum_prob = quantum_prob
        self.crossover_prob = crossover_prob
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_global_position = None
        best_global_value = float('inf')

        positions = self.initialize_positions(lb, ub)
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_values = np.array([float('inf')] * self.num_particles)

        while self.evaluations < self.budget:
            for i in range(self.num_particles):
                if np.random.rand() < self.quantum_prob:
                    positions[i] = self.quantum_perturbation(positions[i], lb, ub)

                velocities[i] = (self.inertia * velocities[i] +
                                 self.cognitive * np.random.random(self.dim) * (personal_best_positions[i] - positions[i]) +
                                 self.social * np.random.random(self.dim) * (best_global_position - positions[i] if best_global_position is not None else 0))
                
                if np.random.rand() < self.crossover_prob:
                    partner_idx = np.random.randint(self.num_particles)
                    crossover_point = np.random.randint(1, self.dim)
                    velocities[i][:crossover_point], velocities[partner_idx][:crossover_point] = (
                        velocities[partner_idx][:crossover_point].copy(), velocities[i][:crossover_point].copy())

                positions[i] = np.clip(positions[i] + velocities[i], lb, ub)

                value = func(positions[i])
                self.evaluations += 1

                if value < personal_best_values[i]:
                    personal_best_values[i] = value
                    personal_best_positions[i] = positions[i]

                if value < best_global_value:
                    best_global_value = value
                    best_global_position = positions[i]

                if self.evaluations >= self.budget:
                    break

        return best_global_position
    
    def initialize_positions(self, lb, ub):
        return np.random.uniform(lb, ub, (self.num_particles, self.dim))

    def quantum_perturbation(self, position, lb, ub):
        q_position = position + (np.random.rand(self.dim) - 0.5) * (ub - lb) * 0.1
        return np.clip(q_position, lb, ub)