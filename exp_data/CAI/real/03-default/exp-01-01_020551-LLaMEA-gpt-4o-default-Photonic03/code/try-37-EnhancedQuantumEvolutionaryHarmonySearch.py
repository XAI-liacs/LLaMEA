import numpy as np

class EnhancedQuantumEvolutionaryHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.min_harmony_memory_size = max(5, dim // 2)
        self.max_harmony_memory_size = max(10, dim)
        self.initial_hmcr = 0.9
        self.initial_par = 0.3
        self.beta = 0.1  # Quantum learning rate
        self.mutation_strength = 0.1
        self.adaptive_rate = 0.002
        self.elite_preservation_rate = 0.05
        self.dynamic_learning_rate = 0.025

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        harmony_memory_size = self.max_harmony_memory_size
        harmony_memory = np.random.uniform(lb, ub, (harmony_memory_size, self.dim))
        scores = np.array([func(harmony_memory[i]) for i in range(harmony_memory_size)])
        global_best_index = np.argmin(scores)
        global_best_position = harmony_memory[global_best_index].copy()
        evaluations = harmony_memory_size
        hmcr = self.initial_hmcr
        par = self.initial_par

        while evaluations < self.budget:
            new_harmony = np.empty(self.dim)
            for i in range(self.dim):
                if np.random.rand() < hmcr:
                    selected = np.random.randint(harmony_memory_size)
                    new_harmony[i] = harmony_memory[selected, i]
                    if np.random.rand() < par:
                        new_harmony[i] += np.random.uniform(-self.mutation_strength, self.mutation_strength) * (ub[i] - lb[i])
                else:
                    new_harmony[i] = np.random.uniform(lb[i], ub[i])

                # Quantum-inspired update with dynamic learning rate
                if np.random.rand() < self.beta:
                    q = np.random.normal(loc=0, scale=1)
                    new_harmony[i] = global_best_position[i] + q * (ub[i] - lb[i]) * self.dynamic_learning_rate

            new_score = func(new_harmony)
            evaluations += 1

            # Elite preservation and evolutionary replacement strategy
            if new_score < scores.max():
                worst_index = np.argmax(scores)
                harmony_memory[worst_index] = new_harmony
                scores[worst_index] = new_score

            if np.random.rand() < self.elite_preservation_rate:
                elite_index = np.random.randint(harmony_memory_size)
                if scores[elite_index] < scores[global_best_index]:
                    global_best_index = elite_index
                    global_best_position = harmony_memory[global_best_index].copy()

            global_best_index = np.argmin(scores)
            global_best_position = harmony_memory[global_best_index].copy()

            # Adaptive parameter tuning
            hmcr = self.initial_hmcr - self.adaptive_rate * (evaluations / self.budget)
            par = self.initial_par + self.adaptive_rate * (evaluations / self.budget)

            # Dynamic memory size adjustment
            if evaluations % (self.budget // 10) == 0:
                harmony_memory_size = self.min_harmony_memory_size + (self.max_harmony_memory_size - self.min_harmony_memory_size) * (1 - evaluations / self.budget)
                harmony_memory_size = int(np.clip(harmony_memory_size, self.min_harmony_memory_size, self.max_harmony_memory_size))
                harmony_memory = np.resize(harmony_memory, (harmony_memory_size, self.dim))
                scores = np.resize(scores, harmony_memory_size)

        return global_best_position, scores[global_best_index]