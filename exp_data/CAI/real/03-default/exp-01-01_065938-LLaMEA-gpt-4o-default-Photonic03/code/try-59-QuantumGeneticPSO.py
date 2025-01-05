import numpy as np

class QuantumGeneticPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.inertia_weight = 0.7
        self.cognitive_comp = 1.5
        self.social_comp = 1.5
        self.mutation_rate = 0.1
        self.quantum_prob = 0.5

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        position = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        velocity = np.random.rand(self.population_size, self.dim) * 0.1
        personal_best_position = np.copy(position)
        personal_best_score = np.array([func(x) for x in personal_best_position])
        global_best_idx = np.argmin(personal_best_score)
        global_best_position = personal_best_position[global_best_idx]

        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocity[i] = (self.inertia_weight * velocity[i] +
                               self.cognitive_comp * r1 * (personal_best_position[i] - position[i]) +
                               self.social_comp * r2 * (global_best_position - position[i]))
                
                position[i] += velocity[i]
                position[i] = np.clip(position[i], lb, ub)

                if np.random.rand() < self.quantum_prob:
                    quantum_bit_flip = np.random.rand(self.dim) < self.mutation_rate
                    position[i][quantum_bit_flip] = lb[quantum_bit_flip] + (ub[quantum_bit_flip] - lb[quantum_bit_flip]) * np.random.rand(len(quantum_bit_flip[quantum_bit_flip]))

                score = func(position[i])
                evaluations += 1

                if score < personal_best_score[i]:
                    personal_best_score[i] = score
                    personal_best_position[i] = position[i]

                    if score < personal_best_score[global_best_idx]:
                        global_best_idx = i
                        global_best_position = position[i]

        return global_best_position