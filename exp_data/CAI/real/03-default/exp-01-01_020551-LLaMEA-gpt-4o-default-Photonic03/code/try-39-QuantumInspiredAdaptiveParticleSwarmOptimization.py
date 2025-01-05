import numpy as np

class QuantumInspiredAdaptiveParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = max(20, 5 * dim)
        self.c1_initial = 2.0  # Cognitive component
        self.c2_initial = 2.0  # Social component
        self.w_initial = 0.9   # Inertia weight
        self.w_final = 0.4
        self.quantum_probability = 0.1

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        swarm_positions = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        swarm_velocities = np.zeros((self.swarm_size, self.dim))
        personal_best_positions = swarm_positions.copy()
        personal_best_scores = np.array([func(swarm_positions[i]) for i in range(self.swarm_size)])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index].copy()
        global_best_score = personal_best_scores[global_best_index]
        evaluations = self.swarm_size

        while evaluations < self.budget:
            c1 = self.c1_initial - evaluations / self.budget * (self.c1_initial - 1.5)
            c2 = self.c2_initial + evaluations / self.budget * (2.5 - self.c2_initial)
            w = self.w_initial - evaluations / self.budget * (self.w_initial - self.w_final)

            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                swarm_velocities[i] = (w * swarm_velocities[i] +
                                       c1 * r1 * (personal_best_positions[i] - swarm_positions[i]) +
                                       c2 * r2 * (global_best_position - swarm_positions[i]))

                # Quantum-inspired position update
                if np.random.rand() < self.quantum_probability:
                    q = np.random.normal(loc=0, scale=1, size=self.dim)
                    swarm_positions[i] = global_best_position + q * (ub - lb)
                else:
                    swarm_positions[i] += swarm_velocities[i]

                swarm_positions[i] = np.clip(swarm_positions[i], lb, ub)
                score = func(swarm_positions[i])
                evaluations += 1

                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = swarm_positions[i].copy()

                if score < global_best_score:
                    global_best_score = score
                    global_best_position = swarm_positions[i].copy()

        return global_best_position, global_best_score