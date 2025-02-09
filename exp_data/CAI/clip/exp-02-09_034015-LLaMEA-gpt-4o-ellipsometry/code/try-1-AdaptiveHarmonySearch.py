import numpy as np

class AdaptiveHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 12  # Increased memory size
        self.hmcr = 0.85  # Adjusted Harmony Memory Considering Rate
        self.par = 0.4   # Adjusted Pitch Adjustment Rate
        self.bandwidth = 0.15  # Adjusted bandwidth

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
                        dynamic_bandwidth = self.bandwidth * np.exp(-0.1 * eval_count / self.budget)
                        new_harmony[i] += dynamic_bandwidth * (np.random.rand() - 0.5) * (ub[i] - lb[i])
                else:
                    new_harmony[i] = np.random.uniform(lb[i], ub[i])

            new_harmony = np.clip(new_harmony, lb, ub)
            new_fitness = func(new_harmony)
            eval_count += 1

            if new_fitness < best_fitness:
                best_harmony = new_harmony.copy()
                best_fitness = new_fitness

            diversity_index = np.std(harmony_memory, axis=0).mean()  # Added diversity measurement
            if new_fitness < fitness[np.argmax(fitness)] or diversity_index < 0.1:
                worst_idx = np.random.choice(np.where(fitness < np.median(fitness))[0])
                harmony_memory[worst_idx, :] = new_harmony
                fitness[worst_idx] = new_fitness

            self._adjust_parameters()

        return best_harmony, best_fitness

    def _adjust_parameters(self):
        self.hmcr = min(1.0, self.hmcr + 0.005)
        self.par = max(0.1, self.par - 0.005)
        self.bandwidth = max(0.05, self.bandwidth - 0.0025)