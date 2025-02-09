import numpy as np

class EnhancedAdaptiveHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 15  # Increased memory size for genetic diversity
        self.hmcr = 0.9  # Slightly increased HMCR for better convergence
        self.par = 0.3   # Reduced PAR for more exploitation
        self.bandwidth = 0.1  # Reduced bandwidth to focus on exploitation

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
                    contributor = np.random.randint(max(1, self.harmony_memory_size // 2))  # Elite selection
                    new_harmony[i] = harmony_memory[contributor, i]
                    if np.random.rand() < self.par:
                        new_harmony[i] += self.bandwidth * (np.random.rand() - 0.5) * (ub[i] - lb[i])
                else:
                    new_harmony[i] = np.random.uniform(lb[i], ub[i])  # Adjusted exploration

            new_harmony = np.clip(new_harmony, lb, ub)
            new_fitness = func(new_harmony)
            eval_count += 1

            if new_fitness < best_fitness:
                best_harmony = new_harmony.copy()
                best_fitness = new_fitness

            worst_idx = np.argmax(fitness)
            if new_fitness < fitness[worst_idx]:
                harmony_memory[worst_idx, :] = new_harmony
                fitness[worst_idx] = new_fitness

            self._adjust_parameters()
            self.harmony_memory_size = min(30, self.harmony_memory_size + 1)  # Dynamic memory size

        return best_harmony, best_fitness

    def _adjust_parameters(self):
        self.hmcr = min(1.0, self.hmcr + 0.01)  # Slower adaptation of HMCR
        self.par = max(0.1, self.par - 0.01)  # Added more gradual reduction of PAR
        self.bandwidth = max(0.05, self.bandwidth * 0.995)  # Slower dynamic bandwidth scaling