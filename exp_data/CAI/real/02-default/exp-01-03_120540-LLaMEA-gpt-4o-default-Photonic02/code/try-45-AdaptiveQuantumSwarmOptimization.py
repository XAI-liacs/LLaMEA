import numpy as np

class AdaptiveQuantumSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.positions = None
        self.best_position = None
        self.best_score = float('inf')
        self.velocities = None
        self.initial_population_size = 30
        self.alpha = 0.5  # quantum-inspired update factor

    def initialize_positions_and_velocities(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))

    def adaptive_population_size(self, evaluations):
        return max(5, int(self.initial_population_size * (1 - evaluations / self.budget)))

    def quantum_swarm_update(self, global_best, individual_best, position):
        quantum_prob = np.random.rand(self.dim)
        position_update = np.where(
            quantum_prob < 0.5,
            position + self.alpha * (individual_best - position),
            position + self.alpha * (global_best - position)
        )
        return position_update

    def __call__(self, func):
        self.initialize_positions_and_velocities(func.bounds)
        evaluations = 0
        individual_best_positions = np.copy(self.positions)
        individual_best_scores = np.full(self.population_size, np.inf)

        while evaluations < self.budget:
            current_population_size = self.adaptive_population_size(evaluations)
            for i in range(current_population_size):
                if evaluations >= self.budget:
                    break

                score = func(self.positions[i])
                evaluations += 1

                if score < individual_best_scores[i]:
                    individual_best_scores[i] = score
                    individual_best_positions[i] = self.positions[i]

                if score < self.best_score:
                    self.best_score = score
                    self.best_position = self.positions[i]

                # Quantum-inspired position update
                self.positions[i] = self.quantum_swarm_update(
                    self.best_position, individual_best_positions[i], self.positions[i])

                # Ensure the new position is within bounds
                self.positions[i] = np.clip(self.positions[i], func.bounds.lb, func.bounds.ub)