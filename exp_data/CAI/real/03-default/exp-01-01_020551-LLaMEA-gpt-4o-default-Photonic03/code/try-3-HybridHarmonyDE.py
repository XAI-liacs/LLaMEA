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
        self.evaluations = 0

    def adapt_parameters(self):
        # Dynamically adapt parameters based on the ratio of evaluations to budget
        progress = self.evaluations / self.budget
        self.hmcr = 0.9 - 0.2 * progress  # Decrease hmcr to promote exploration
        self.par = 0.3 + 0.4 * progress   # Increase par to promote exploitation
        self.f = 0.8 + 0.2 * progress     # Slightly increase differential weight for later stages
        self.cr = 0.9 - 0.3 * progress    # Decrease crossover probability for diversity

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        harmony_memory = np.random.uniform(lb, ub, (self.harmony_memory_size, self.dim))
        scores = np.array([func(harmony_memory[i]) for i in range(self.harmony_memory_size)])
        self.evaluations = self.harmony_memory_size

        while self.evaluations < self.budget:
            self.adapt_parameters()

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
            if self.evaluations + 3 < self.budget:
                indices = np.random.choice(self.harmony_memory_size, 3, replace=False)
                x0, x1, x2 = harmony_memory[indices]
                mutant = np.clip(x0 + self.f * (x1 - x2), lb, ub)
                trial = np.where(np.random.rand(self.dim) < self.cr, mutant, new_harmony)
            else:
                trial = new_harmony

            new_score = func(trial)
            self.evaluations += 1

            # Update harmony memory
            if new_score < scores.max():
                worst_index = np.argmax(scores)
                harmony_memory[worst_index] = trial
                scores[worst_index] = new_score

        # Return the best solution found
        best_index = np.argmin(scores)
        return harmony_memory[best_index], scores[best_index]