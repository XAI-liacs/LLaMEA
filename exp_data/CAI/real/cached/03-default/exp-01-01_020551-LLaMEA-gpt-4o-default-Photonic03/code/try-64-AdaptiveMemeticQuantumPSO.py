import numpy as np

class AdaptiveMemeticQuantumPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, dim)
        self.inertia_weight = 0.7
        self.cognitive_const = 1.5
        self.social_const = 1.5
        self.quantum_prob = 0.3
        self.local_search_prob = 0.1
        self.search_strength = 0.05

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        position = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocity = np.random.uniform(-1, 1, (self.population_size, self.dim)) * (ub - lb) / 10
        personal_best_position = position.copy()
        personal_best_score = np.array([func(pos) for pos in position])
        global_best_index = np.argmin(personal_best_score)
        global_best_position = personal_best_position[global_best_index].copy()
        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                velocity[i] = (self.inertia_weight * velocity[i]
                               + self.cognitive_const * r1 * (personal_best_position[i] - position[i])
                               + self.social_const * r2 * (global_best_position - position[i]))
                position[i] += velocity[i]
                position[i] = np.clip(position[i], lb, ub)

                # Quantum-inspired perturbation
                if np.random.rand() < self.quantum_prob:
                    position[i] += np.random.normal(0, 0.1, self.dim) * (global_best_position - position[i])

                # Local search enhancement
                if np.random.rand() < self.local_search_prob:
                    neighbor = position[i] + np.random.uniform(-self.search_strength, self.search_strength, self.dim) * (ub - lb)
                    neighbor = np.clip(neighbor, lb, ub)
                    neighbor_score = func(neighbor)
                    evaluations += 1
                    if neighbor_score < personal_best_score[i]:
                        personal_best_position[i] = neighbor
                        personal_best_score[i] = neighbor_score

                new_score = func(position[i])
                evaluations += 1

                if new_score < personal_best_score[i]:
                    personal_best_position[i] = position[i].copy()
                    personal_best_score[i] = new_score

            global_best_index = np.argmin(personal_best_score)
            global_best_position = personal_best_position[global_best_index].copy()

        return global_best_position, personal_best_score[global_best_index]