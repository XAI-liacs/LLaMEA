import numpy as np

class GaussianHarmonyEvolutionStrategy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = max(10, dim)
        self.hmcr = 0.9  # Harmony Memory Consideration Rate
        self.par = 0.3   # Pitch Adjustment Rate
        self.gaussian_sigma = 0.1
        self.local_search_rate = 0.2

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        harmony_memory = np.random.uniform(lb, ub, (self.harmony_memory_size, self.dim))
        scores = np.array([func(harmony_memory[i]) for i in range(self.harmony_memory_size)])
        evaluations = self.harmony_memory_size

        while evaluations < self.budget:
            new_harmony = np.empty(self.dim)
            for i in range(self.dim):
                if np.random.rand() < self.hmcr:
                    new_harmony[i] = harmony_memory[np.random.randint(self.harmony_memory_size), i]
                    if np.random.rand() < self.par:
                        new_harmony[i] += np.random.normal(0, self.gaussian_sigma) * (ub[i] - lb[i])
                else:
                    new_harmony[i] = np.random.uniform(lb[i], ub[i])
            
            new_harmony = np.clip(new_harmony, lb, ub)

            # Local search strategy
            if np.random.rand() < self.local_search_rate:
                local_harmony = new_harmony + np.random.normal(0, self.gaussian_sigma, self.dim)
                local_harmony = np.clip(local_harmony, lb, ub)
                local_score = func(local_harmony)
                evaluations += 1
                if local_score < func(new_harmony):
                    new_harmony = local_harmony

            new_score = func(new_harmony)
            evaluations += 1

            # Update harmony memory
            if new_score < scores.max():
                worst_index = np.argmax(scores)
                harmony_memory[worst_index] = new_harmony
                scores[worst_index] = new_score

        best_index = np.argmin(scores)
        return harmony_memory[best_index], scores[best_index]