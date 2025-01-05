import numpy as np

class SwarmEnhancedQuantumAdaptiveNeighborhoodSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, dim)
        self.quantum_rate = 0.3
        self.swarm_influence = 0.5
        self.velocity_weight = 0.7
        self.local_attraction = 1.5
        self.global_attraction = 1.5
        self.adaptive_rate = 0.01

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        scores = np.array([func(population[i]) for i in range(self.population_size)])
        personal_best_positions = population.copy()
        personal_best_scores = scores.copy()
        global_best_index = np.argmin(scores)
        global_best_position = population[global_best_index].copy()
        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Swarm-based velocity update
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.velocity_weight * velocities[i] +
                                 self.local_attraction * r1 * (personal_best_positions[i] - population[i]) +
                                 self.global_attraction * r2 * (global_best_position - population[i]))

                # Update position
                population[i] += velocities[i]
                population[i] = np.clip(population[i], lb, ub)

                # Quantum-inspired mutation
                if np.random.rand() < self.quantum_rate:
                    q = np.random.normal(0, 0.5, self.dim)
                    population[i] += q * (ub - lb)

                # Evaluate new position
                new_score = func(population[i])
                evaluations += 1

                # Update personal best
                if new_score < personal_best_scores[i]:
                    personal_best_scores[i] = new_score
                    personal_best_positions[i] = population[i].copy()

                # Update global best
                if new_score < personal_best_scores[global_best_index]:
                    global_best_index = i
                    global_best_position = population[i].copy()

            # Adaptive parameter adjustment
            self.quantum_rate = 0.3 - self.adaptive_rate * (evaluations / self.budget)
            self.velocity_weight = 0.7 - self.adaptive_rate * (evaluations / self.budget)

        return global_best_position, personal_best_scores[global_best_index]