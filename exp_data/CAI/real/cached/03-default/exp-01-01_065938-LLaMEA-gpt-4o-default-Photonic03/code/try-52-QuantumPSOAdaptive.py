import numpy as np

class QuantumPSOAdaptive:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 50
        self.c1 = 2.0
        self.c2 = 2.0
        self.w = 0.5
        self.quantum_factor = 0.5
        self.entropy_factor = 0.1
        self.velocity_clamp = 0.1

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        
        # Initialize positions and velocities
        swarm = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-self.velocity_clamp, self.velocity_clamp, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(swarm)
        personal_best_scores = np.array([func(x) for x in swarm])
        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_score = personal_best_scores[global_best_idx]

        evaluations = self.swarm_size

        while evaluations < self.budget:
            for i in range(self.swarm_size):
                # Update velocity
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive = self.c1 * r1 * (personal_best_positions[i] - swarm[i])
                social = self.c2 * r2 * (global_best_position - swarm[i])
                velocities[i] = (self.w * velocities[i] + cognitive + social +
                                 self.entropy_factor * np.random.uniform(-1, 1, self.dim))

                # Clamp velocity
                velocities[i] = np.clip(velocities[i], -self.velocity_clamp, self.velocity_clamp)

                # Update position
                swarm[i] += velocities[i]
                
                # Quantum-inspired update
                quantum_jump = self.quantum_factor * np.random.uniform(lb, ub, self.dim)
                swarm[i] = (swarm[i] + np.sin(np.pi * quantum_jump)) / 2
                
                swarm[i] = np.clip(swarm[i], lb, ub)

                # Evaluate new position
                score = func(swarm[i])
                evaluations += 1
                
                # Update personal best
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = np.copy(swarm[i])

                    # Update global best
                    if score < global_best_score:
                        global_best_score = score
                        global_best_position = np.copy(swarm[i])

            # Adjust parameters
            self.w = np.clip(self.w + 0.05 * (np.random.rand() - 0.5), 0.3, 0.7)
            self.entropy_factor = np.clip(self.entropy_factor + 0.01 * (np.random.rand() - 0.5), 0.05, 0.15)
            self.quantum_factor = np.clip(self.quantum_factor + 0.01 * (np.random.rand() - 0.5), 0.3, 0.7)

        return global_best_position