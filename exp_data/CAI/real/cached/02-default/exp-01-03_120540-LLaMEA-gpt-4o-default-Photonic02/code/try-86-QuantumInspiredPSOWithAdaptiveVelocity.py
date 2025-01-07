import numpy as np

class QuantumInspiredPSOWithAdaptiveVelocity:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.positions = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.inertia_weight = 0.7
        self.cognitive_coefficient = 1.5
        self.social_coefficient = 1.5
    
    def initialize_positions(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.population_size, np.inf)
    
    def quantum_inspired_update(self, velocity, position, bounds):
        lb, ub = bounds.lb, bounds.ub
        quantum_bit = np.random.rand(self.dim) < 0.5
        quantum_flip = np.where(quantum_bit, 1, -1)
        new_velocity = velocity * quantum_flip
        new_position = position + new_velocity
        new_position = np.clip(new_position, lb, ub)
        return new_position, new_velocity
    
    def update_velocity_and_position(self, idx, bounds):
        r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
        cognitive_velocity = self.cognitive_coefficient * r1 * (self.personal_best_positions[idx] - self.positions[idx])
        social_velocity = self.social_coefficient * r2 * (self.global_best_position - self.positions[idx])
        self.velocities[idx] = (self.inertia_weight * self.velocities[idx] + cognitive_velocity + social_velocity)
        
        # Adaptive velocity control
        self.velocities[idx] *= (1 - np.linalg.norm(self.positions[idx] - self.global_best_position) / np.linalg.norm(bounds.ub - bounds.lb))
        
        self.positions[idx] += self.velocities[idx]
        self.positions[idx] = np.clip(self.positions[idx], bounds.lb, bounds.ub)
    
    def __call__(self, func):
        self.initialize_positions(func.bounds)
        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                current_score = func(self.positions[i])
                evaluations += 1

                if current_score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = current_score
                    self.personal_best_positions[i] = np.copy(self.positions[i])

                if current_score < self.global_best_score:
                    self.global_best_score = current_score
                    self.global_best_position = np.copy(self.positions[i])

                self.update_velocity_and_position(i, func.bounds)

        # Quantum-inspired global search for final optimization
        for i in range(self.population_size):
            if evaluations >= self.budget:
                break

            new_position, new_velocity = self.quantum_inspired_update(self.velocities[i], self.positions[i], func.bounds)
            new_score = func(new_position)
            evaluations += 1

            if new_score < self.global_best_score:
                self.global_best_score = new_score
                self.global_best_position = new_position