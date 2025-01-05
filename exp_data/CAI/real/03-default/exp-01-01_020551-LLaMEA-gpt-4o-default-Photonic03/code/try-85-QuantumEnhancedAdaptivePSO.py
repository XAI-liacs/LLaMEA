import numpy as np

class QuantumEnhancedAdaptivePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(20, dim * 2)
        self.inertia_weight = 0.7  # Initial inertia weight
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.quantum_coeff = 0.2  # Quantum enhancement factor
        self.adaptive_rate = 0.01  # Adaptive parameter change rate

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-abs(ub - lb), abs(ub - lb), (self.population_size, self.dim))
        personal_best_positions = positions.copy()
        personal_best_scores = np.array([func(x) for x in positions])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index].copy()
        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.inertia_weight * velocities[i]
                                 + self.cognitive_coeff * r1 * (personal_best_positions[i] - positions[i])
                                 + self.social_coeff * r2 * (global_best_position - positions[i]))
                # Quantum-inspired update
                if np.random.rand() < self.quantum_coeff:
                    q = np.random.normal(loc=0, scale=0.5)
                    velocities[i] += q * (ub - lb)

                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], lb, ub)

                score = func(positions[i])
                evaluations += 1

                if score < personal_best_scores[i]:
                    personal_best_positions[i] = positions[i]
                    personal_best_scores[i] = score

                    if score < personal_best_scores[global_best_index]:
                        global_best_index = i
                        global_best_position = positions[i]

            # Adapt inertia weight based on evaluations
            self.inertia_weight = 0.9 - (0.5 * evaluations / self.budget)

        return global_best_position, personal_best_scores[global_best_index]