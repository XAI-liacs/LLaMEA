import numpy as np

class QuantumEnhancedHarmonyPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = max(20, dim)
        self.hmcr = 0.9
        self.par = 0.3
        self.inertia = 0.7
        self.c1 = 1.5
        self.c2 = 1.5
        self.beta = 0.05

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        harmony_memory = np.random.uniform(lb, ub, (self.harmony_memory_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.harmony_memory_size, self.dim)) * (ub - lb)
        temp_harmony_memory = harmony_memory.copy()
        personal_best_positions = harmony_memory.copy()
        personal_best_scores = np.array([func(harmony_memory[i]) for i in range(self.harmony_memory_size)])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = harmony_memory[global_best_index].copy()
        evaluations = self.harmony_memory_size

        while evaluations < self.budget:
            self.hmcr = 0.7 + 0.3 * (1 - evaluations / self.budget)
            self.par = 0.2 + 0.5 * (evaluations / self.budget)

            for i in range(self.harmony_memory_size):
                new_harmony = np.empty(self.dim)
                for d in range(self.dim):
                    if np.random.rand() < self.hmcr:
                        new_harmony[d] = harmony_memory[np.random.randint(self.harmony_memory_size), d]
                        if np.random.rand() < self.par:
                            new_harmony[d] += np.random.uniform(-0.1, 0.1) * (ub[d] - lb[d])
                    else:
                        new_harmony[d] = np.random.uniform(lb[d], ub[d])

                    if np.random.rand() < self.beta:
                        q = np.random.normal(loc=0, scale=1)
                        new_harmony[d] = global_best_position[d] + q * (ub[d] - lb[d])

                new_score = func(new_harmony)
                evaluations += 1

                if new_score < personal_best_scores[i]:
                    personal_best_positions[i] = new_harmony
                    personal_best_scores[i] = new_score
                    if new_score < personal_best_scores[global_best_index]:
                        global_best_index = i
                        global_best_position = new_harmony

            r1, r2 = np.random.rand(self.harmony_memory_size, self.dim), np.random.rand(self.harmony_memory_size, self.dim)
            velocities = (self.inertia * velocities +
                          self.c1 * r1 * (personal_best_positions - harmony_memory) +
                          self.c2 * r2 * (global_best_position - harmony_memory))
            temp_harmony_memory = np.clip(harmony_memory + velocities, lb, ub)

            for i in range(self.harmony_memory_size):
                temp_score = func(temp_harmony_memory[i])
                evaluations += 1
                if temp_score < personal_best_scores[i]:
                    harmony_memory[i] = temp_harmony_memory[i]
                    personal_best_scores[i] = temp_score
                    if temp_score < personal_best_scores[global_best_index]:
                        global_best_index = i
                        global_best_position = temp_harmony_memory[i]

            if evaluations >= self.budget:
                break

        return global_best_position, personal_best_scores[global_best_index]