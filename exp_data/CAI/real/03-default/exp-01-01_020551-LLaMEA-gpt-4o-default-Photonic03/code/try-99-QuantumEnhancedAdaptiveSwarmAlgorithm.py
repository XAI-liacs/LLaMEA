import numpy as np

class QuantumEnhancedAdaptiveSwarmAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = max(20, dim * 5)
        self.alpha = 0.5  # Initial weight for exploitation
        self.beta = 0.3   # Initial weight for exploration
        self.gamma = 0.1  # Quantum influence factor
        self.mutation_strength = 0.05
        self.adaptive_factor = 0.01  # Adaptive adjustment rate

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        swarm = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-0.1, 0.1, (self.swarm_size, self.dim))
        scores = np.array([func(p) for p in swarm])
        global_best_index = np.argmin(scores)
        global_best_position = swarm[global_best_index].copy()
        evaluations = self.swarm_size

        while evaluations < self.budget:
            for i in range(self.swarm_size):
                # Update velocities with quantum-inspired perturbation
                velocities[i] = (self.alpha * velocities[i]
                                + self.beta * (global_best_position - swarm[i])
                                + self.gamma * np.random.normal(0, 1, self.dim) * (ub - lb))
                
                # Update positions
                swarm[i] += velocities[i]
                swarm[i] = np.clip(swarm[i], lb, ub)

                # Quantum-inspired mutation
                if np.random.rand() < self.gamma:
                    swarm[i] += np.random.normal(0, self.mutation_strength, self.dim) * (ub - lb)
                    swarm[i] = np.clip(swarm[i], lb, ub)

                # Evaluate new position
                new_score = func(swarm[i])
                evaluations += 1
                if new_score < scores[i]:
                    scores[i] = new_score

                # Update global best
                if new_score < scores[global_best_index]:
                    global_best_index = i
                    global_best_position = swarm[i].copy()

            # Adaptive parameter tuning
            self.alpha += self.adaptive_factor * (1 - evaluations / self.budget)
            self.beta -= self.adaptive_factor * (1 - evaluations / self.budget)
            self.gamma *= 1 + self.adaptive_factor * np.sin(2 * np.pi * evaluations / self.budget)

        return global_best_position, scores[global_best_index]