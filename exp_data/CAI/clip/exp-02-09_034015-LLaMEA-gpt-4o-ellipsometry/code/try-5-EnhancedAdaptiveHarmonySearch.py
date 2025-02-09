import numpy as np

class EnhancedAdaptiveHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 10 + int(dim / 2)  # Dynamic memory size based on dimension
        self.hmcr = 0.85
        self.par = 0.35
        self.bandwidth = 0.2  # Adjusted bandwidth for diversity

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        harmony_memory = np.random.uniform(lb, ub, (self.harmony_memory_size, self.dim))
        fitness = np.apply_along_axis(func, 1, harmony_memory)

        eval_count = self.harmony_memory_size
        best_idx = np.argmin(fitness)
        best_harmony = harmony_memory[best_idx].copy()
        best_fitness = fitness[best_idx]

        while eval_count < self.budget:
            new_harmony = np.zeros(self.dim)
            for i in range(self.dim):
                if np.random.rand() < self.hmcr:
                    contributor = np.random.randint(self.harmony_memory_size)
                    new_harmony[i] = harmony_memory[contributor, i]
                    if np.random.rand() < self.par:
                        new_harmony[i] += self.bandwidth * np.random.normal(0, 1) * (ub[i] - lb[i])  # Adjusted randomness
                else:
                    new_harmony[i] = np.random.normal((ub[i] + lb[i]) / 2, (ub[i] - lb[i]) / 2)  # Centralized normal distribution

            new_harmony = np.clip(new_harmony, lb, ub)
            new_fitness = func(new_harmony)
            eval_count += 1

            if new_fitness < best_fitness:
                best_harmony = new_harmony.copy()
                best_fitness = new_fitness

            if new_fitness < fitness.max():
                worst_idx = np.argmax(fitness)
                harmony_memory[worst_idx, :] = new_harmony
                fitness[worst_idx] = new_fitness

            self._adjust_parameters()

        return best_harmony, best_fitness

    def _adjust_parameters(self):
        self.hmcr = min(1.0, self.hmcr + 0.01)  # Adjusted adaptation of HMCR
        self.par = max(0.15, self.par - 0.005)  # Adjusted reduction of PAR
        self.bandwidth = max(0.1, self.bandwidth - 0.005)  # Adjusted reduction of bandwidth