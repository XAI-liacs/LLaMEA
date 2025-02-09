import numpy as np

class EnhancedAdaptiveHarmonySearch:  # (1)
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 10
        self.hmcr = 0.9
        self.par = 0.3
        self.bandwidth = 0.1
        self.elitism_rate = 0.2  # (2)

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
                        new_harmony[i] += self.bandwidth * (np.random.rand() - 0.5) * (ub[i] - lb[i])
                else:
                    new_harmony[i] = np.random.uniform(lb[i], ub[i])

            new_harmony = np.clip(new_harmony, lb, ub)
            new_fitness = func(new_harmony)
            eval_count += 1

            if new_fitness < best_fitness:
                best_harmony = new_harmony.copy()
                best_fitness = new_fitness

            self._update_harmony_memory(new_harmony, new_fitness, fitness, harmony_memory)  # (3)

            self._adjust_parameters()

        return best_harmony, best_fitness

    def _adjust_parameters(self):
        self.hmcr = min(1.0, self.hmcr + 0.01)
        self.par = max(0.1, self.par - 0.01)
        self.bandwidth = max(0.05, self.bandwidth * 0.99)  # (4)

    def _update_harmony_memory(self, new_harmony, new_fitness, fitness, harmony_memory):  # (5)
        sorted_indices = np.argsort(fitness)
        worst_idx = sorted_indices[-int(self.elitism_rate * self.harmony_memory_size)]  # (6)
        if new_fitness < fitness[worst_idx]:
            harmony_memory[worst_idx, :] = new_harmony
            fitness[worst_idx] = new_fitness