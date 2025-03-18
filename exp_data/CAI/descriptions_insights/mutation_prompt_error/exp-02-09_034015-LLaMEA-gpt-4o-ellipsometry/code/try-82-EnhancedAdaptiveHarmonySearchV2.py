import numpy as np

class EnhancedAdaptiveHarmonySearchV2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 15
        self.hmcr = 0.9
        self.par = 0.3
        self.bandwidth = 0.1
        self.success_rate_threshold = 0.2
        self.adaptive_factor = 0.01

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        harmony_memory = np.random.uniform(lb, ub, (self.harmony_memory_size, self.dim))
        fitness = np.apply_along_axis(func, 1, harmony_memory)
        
        eval_count = self.harmony_memory_size
        best_idx = np.argmin(fitness)
        best_harmony = harmony_memory[best_idx].copy()
        best_fitness = fitness[best_idx]
        success_count = 0

        while eval_count < self.budget:
            new_harmony = np.zeros(self.dim)
            for i in range(self.dim):
                if np.random.rand() < self.hmcr:
                    contributor = np.random.randint(self.harmony_memory_size)
                    new_harmony[i] = harmony_memory[contributor, i]
                    if np.random.rand() < self.par:
                        new_harmony[i] += self.bandwidth * (np.random.rand() - 0.5) * (ub[i] - lb[i])
                else:
                    new_harmony[i] = np.random.uniform(lb[i], ub[i])

            new_harmony = np.clip(new_harmony, lb, ub)
            new_fitness = func(new_harmony)
            eval_count += 1

            if new_fitness < best_fitness:
                best_harmony = new_harmony.copy()
                best_fitness = new_fitness
                success_count += 1

            worst_idx = np.argmax(fitness)
            if new_fitness < fitness[worst_idx]:
                harmony_memory[worst_idx, :] = new_harmony
                fitness[worst_idx] = new_fitness

            if eval_count % self.harmony_memory_size == 0:
                self._adjust_parameters(success_count / self.harmony_memory_size)
                success_count = 0

        return best_harmony, best_fitness

    def _adjust_parameters(self, success_rate):
        if success_rate < self.success_rate_threshold:
            self.hmcr = min(1.0, self.hmcr + self.adaptive_factor)
            self.par = max(0.1, self.par - self.adaptive_factor)
            self.bandwidth = max(0.05, self.bandwidth * 0.995)
        else:
            self.hmcr = max(0.8, self.hmcr - self.adaptive_factor)
            self.par = min(0.5, self.par + self.adaptive_factor)
            self.bandwidth = min(0.2, self.bandwidth * 1.005)