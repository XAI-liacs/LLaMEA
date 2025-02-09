import numpy as np

class QuantumInspiredPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

        # QIPSO parameters
        self.num_particles = 30
        self.inertia_weight = 0.9
        self.cognitive_param = 1.7
        self.social_param = 1.5
        self.quantum_probability = 0.1

    def __call__(self, func):
        num_evaluations = 0
        bounds = func.bounds
        lb = bounds.lb
        ub = bounds.ub

        # Initialize particles
        positions = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([float('inf')] * self.num_particles)
        
        # Evaluate initial solutions
        for i in range(self.num_particles):
            score = func(positions[i])
            num_evaluations += 1
            personal_best_scores[i] = score
            if num_evaluations >= self.budget:
                return positions[i]
        
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = min(personal_best_scores)

        # Main loop
        while num_evaluations < self.budget:
            for i in range(self.num_particles):
                # Quantum rotation gate application
                if np.random.rand() < self.quantum_probability:
                    rotation_angle = np.random.uniform(-np.pi, np.pi, self.dim)
                    positions[i] = self.apply_quantum_rotation(positions[i], rotation_angle, lb, ub)

                # Update velocity and position
                r1, r2 = np.random.rand(), np.random.rand()
                self.inertia_weight *= 0.99
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 self.cognitive_param * r1 * (personal_best_positions[i] - positions[i]) +
                                 self.social_param * r2 * (global_best_position - positions[i]))
                positions[i] = np.clip(positions[i] + velocities[i], lb, ub)

                # Evaluate and update personal best
                score = func(positions[i])
                num_evaluations += 1
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i]

                # Update global best
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[i]

                if num_evaluations >= self.budget:
                    return global_best_position

        return global_best_position

    def apply_quantum_rotation(self, position, rotation_angle, lb, ub):
        # Apply a quantum-inspired rotation to the position
        new_position = position * np.cos(rotation_angle) + np.sin(rotation_angle)
        return np.clip(new_position, lb, ub)