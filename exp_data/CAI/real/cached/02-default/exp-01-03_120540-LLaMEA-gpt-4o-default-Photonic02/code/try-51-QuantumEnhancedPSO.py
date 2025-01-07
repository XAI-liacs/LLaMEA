import numpy as np

class QuantumEnhancedPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 30
        self.inertia_weight = 0.7
        self.cognitive_coef = 1.5
        self.social_coef = 1.5
        self.positions = np.random.rand(self.swarm_size, self.dim)
        self.velocities = np.random.rand(self.swarm_size, self.dim)
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.swarm_size, float('inf'))
        self.global_best_position = None
        self.global_best_score = float('inf')

    def initialize_positions_and_velocities(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.positions = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        vel_range = (ub - lb) * 0.1
        self.velocities = np.random.uniform(-vel_range, vel_range, (self.swarm_size, self.dim))

    def quantum_behavior(self, position):
        # Simulate quantum behavior by introducing random phase shift
        random_phase = np.random.uniform(-np.pi, np.pi, self.dim)
        quantum_shift = np.sin(random_phase)
        return position + quantum_shift
    
    def update_velocity(self, idx, func):
        r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
        cognitive_component = self.cognitive_coef * r1 * (self.personal_best_positions[idx] - self.positions[idx])
        social_component = self.social_coef * r2 * (self.global_best_position - self.positions[idx])
        self.velocities[idx] = (self.inertia_weight * self.velocities[idx] 
                                + cognitive_component + social_component)
        velocity_clipping_bound = (func.bounds.ub - func.bounds.lb) * 0.2
        self.velocities[idx] = np.clip(self.velocities[idx], -velocity_clipping_bound, velocity_clipping_bound)

    def __call__(self, func):
        self.initialize_positions_and_velocities(func.bounds)
        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.swarm_size):
                if evaluations >= self.budget:
                    break

                self.update_velocity(i, func)
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], func.bounds.lb, func.bounds.ub)

                current_score = func(self.positions[i])
                evaluations += 1

                if current_score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = current_score
                    self.personal_best_positions[i] = self.positions[i]

                if current_score < self.global_best_score:
                    self.global_best_score = current_score
                    self.global_best_position = self.positions[i]
            
            # Quantum-enhanced global search
            for i in range(self.swarm_size):
                if evaluations >= self.budget:
                    break

                quantum_position = self.quantum_behavior(self.positions[i])
                quantum_position = np.clip(quantum_position, func.bounds.lb, func.bounds.ub)
                quantum_score = func(quantum_position)
                evaluations += 1

                if quantum_score < self.global_best_score:
                    self.global_best_score = quantum_score
                    self.global_best_position = quantum_position