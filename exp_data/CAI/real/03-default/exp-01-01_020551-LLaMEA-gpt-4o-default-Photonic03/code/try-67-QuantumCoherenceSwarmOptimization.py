import numpy as np

class QuantumCoherenceSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = max(30, dim * 2)
        self.alpha = 0.1  # Exploration factor
        self.beta = 0.9  # Exploitation factor
        self.gamma = 0.1  # Quantum coherence factor
        self.loudness = 0.5  # Loudness of particles
        self.frequency = 0.5  # Frequency of local search adjustments

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        swarm = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim)) * (ub - lb)
        scores = np.array([func(p) for p in swarm])
        personal_best_positions = swarm.copy()
        personal_best_scores = scores.copy()
        global_best_index = np.argmin(scores)
        global_best_position = swarm[global_best_index].copy()
        evaluations = self.swarm_size

        while evaluations < self.budget:
            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(2)
                velocities[i] = (self.alpha * velocities[i] +
                                 self.beta * r1 * (personal_best_positions[i] - swarm[i]) +
                                 self.beta * r2 * (global_best_position - swarm[i]))
                
                # Quantum coherence update
                if np.random.rand() < self.gamma:
                    coherence_factor = np.random.normal(0, self.frequency, self.dim)
                    velocities[i] += coherence_factor * (global_best_position - personal_best_positions[i])

                swarm[i] += velocities[i]
                swarm[i] = np.clip(swarm[i], lb, ub)

                new_score = func(swarm[i])
                evaluations += 1

                # Update personal and global bests
                if new_score < personal_best_scores[i]:
                    personal_best_positions[i] = swarm[i].copy()
                    personal_best_scores[i] = new_score

                if new_score < scores[global_best_index]:
                    global_best_index = i
                    global_best_position = swarm[i].copy()

            # Adaptive strategy for quantum coherence
            self.gamma = max(0.1, self.gamma - 0.01 * (evaluations / self.budget))
            self.loudness = max(0.1, self.loudness * 0.99)

        return global_best_position, scores[global_best_index]