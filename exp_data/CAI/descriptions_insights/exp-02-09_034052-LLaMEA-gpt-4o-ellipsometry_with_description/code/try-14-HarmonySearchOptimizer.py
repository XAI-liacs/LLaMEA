import numpy as np

class HarmonySearchOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 5
        self.harmony_memory = []
        self.harmony_memory_consideration_rate = 0.9
        self.pitch_adjustment_rate = 0.3
        self.bandwidth = 0.1

    def initialize_harmony_memory(self, lb, ub):
        for _ in range(self.harmony_memory_size):
            harmony = np.random.uniform(lb, ub, self.dim)
            score = float('inf')
            self.harmony_memory.append((harmony, score))

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_harmony_memory(lb, ub)
        evaluations = 0
        best_solution = None
        best_score = float('inf')

        while evaluations < self.budget:
            new_harmony = np.zeros(self.dim)

            for i in range(self.dim):
                if np.random.rand() < self.harmony_memory_consideration_rate:
                    selected_harmony = self.harmony_memory[np.random.randint(self.harmony_memory_size)][0]
                    new_harmony[i] = selected_harmony[i]

                    if np.random.rand() < self.pitch_adjustment_rate:
                        self.bandwidth *= 0.99  # Adaptive bandwidth adjustment
                        new_harmony[i] += self.bandwidth * (np.random.rand() - 0.5) * (ub[i] - lb[i])
                else:
                    new_harmony[i] = np.random.uniform(lb[i], ub[i])
                    
            if np.random.rand() < 0.01:  # Global random search chance
                new_harmony = np.random.uniform(lb, ub, self.dim)

            new_harmony = np.clip(new_harmony, lb, ub)
            new_score = func(new_harmony)
            evaluations += 1

            worst_index = np.argmax([score for _, score in self.harmony_memory])
            if new_score < self.harmony_memory[worst_index][1]:
                self.harmony_memory[worst_index] = (new_harmony, new_score)

            if new_score < best_score:
                best_solution = new_harmony
                best_score = new_score

        return best_solution