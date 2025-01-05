import numpy as np

class HybridHarmonyDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = max(10, dim)
        self.hmcr = 0.9  # Harmony Memory Consideration Rate
        self.par = 0.3   # Pitch Adjustment Rate
        self.f = 0.8     # Differential weight
        self.cr = 0.9    # Crossover probability

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        harmony_memory = np.random.uniform(lb, ub, (self.harmony_memory_size, self.dim))
        scores = np.array([func(harmony_memory[i]) for i in range(self.harmony_memory_size)])
        evaluations = self.harmony_memory_size

        while evaluations < self.budget:
            # Generate new harmony vector
            new_harmony = np.empty(self.dim)
            for i in range(self.dim):
                if np.random.rand() < self.hmcr:
                    new_harmony[i] = harmony_memory[np.random.randint(self.harmony_memory_size), i]
                    if np.random.rand() < self.par:
                        new_harmony[i] += np.random.uniform(-0.1, 0.1) * (ub[i] - lb[i])
                else:
                    new_harmony[i] = np.random.uniform(lb[i], ub[i])

            # Differential Evolution mutation and crossover
            if evaluations + 3 < self.budget:
                indices = np.random.choice(self.harmony_memory_size, 3, replace=False)
                x0, x1, x2 = harmony_memory[indices]
                mutant = np.clip(x0 + self.f * (x1 - x2), lb, ub)
                trial = np.where(np.random.rand(self.dim) < self.cr, mutant, new_harmony)
            else:
                trial = new_harmony

            new_score = func(trial)
            evaluations += 1

            # Update harmony memory
            if new_score < scores.max():
                worst_index = np.argmax(scores)
                harmony_memory[worst_index] = trial
                scores[worst_index] = new_score

        # Return the best solution found
        best_index = np.argmin(scores)
        return harmony_memory[best_index], scores[best_index]