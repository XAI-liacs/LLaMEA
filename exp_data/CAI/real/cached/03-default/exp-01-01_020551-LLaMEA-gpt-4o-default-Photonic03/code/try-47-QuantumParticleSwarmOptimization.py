import numpy as np

class QuantumParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = max(20, 2 * dim)
        self.omega = 0.5  # Inertia weight
        self.phi_p = 0.5  # Cognitive coefficient
        self.phi_g = 0.5  # Social coefficient
        self.alpha = 0.75  # Constriction factor
        self.quantum_delta = 0.1  # Step size in quantum behavior

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pos = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        vel = np.zeros((self.swarm_size, self.dim))
        personal_best_pos = pos.copy()
        personal_best_scores = np.array([func(x) for x in pos])
        global_best_index = np.argmin(personal_best_scores)
        global_best_pos = pos[global_best_index].copy()
        global_best_score = personal_best_scores[global_best_index]
        evaluations = self.swarm_size

        while evaluations < self.budget:
            for i in range(self.swarm_size):
                r_p = np.random.rand(self.dim)
                r_g = np.random.rand(self.dim)

                # Update velocity with quantum behavior
                vel[i] = (self.omega * vel[i] +
                          self.phi_p * r_p * (personal_best_pos[i] - pos[i]) +
                          self.phi_g * r_g * (global_best_pos - pos[i]))

                pos[i] += self.alpha * vel[i] + self.quantum_delta * np.random.normal(size=self.dim) * (ub - lb)
                pos[i] = np.clip(pos[i], lb, ub)

                score = func(pos[i])
                evaluations += 1

                # Update personal best
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_pos[i] = pos[i].copy()

                # Update global best
                if score < global_best_score:
                    global_best_score = score
                    global_best_pos = pos[i].copy()

        return global_best_pos, global_best_score