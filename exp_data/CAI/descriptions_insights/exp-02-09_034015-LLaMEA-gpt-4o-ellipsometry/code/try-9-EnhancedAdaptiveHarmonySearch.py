import numpy as np

class EnhancedAdaptiveHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 12
        self.hmcr = 0.85
        self.par = 0.35
        self.bandwidth = 0.15

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
                        new_harmony[i] += np.random.normal(0, self.bandwidth) * (ub[i] - lb[i])  # Stochastic perturbation
                else:
                    new_harmony[i] = np.random.uniform(lb[i], ub[i])  # Uniform exploration

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

        self._adjust_parameters(eval_count)

        return best_harmony, best_fitness

    def _adjust_parameters(self, eval_count):
        self.hmcr = min(1.0, 0.85 + 0.15 * (eval_count / self.budget))  # Dynamic HMCR
        self.par = max(0.1, 0.35 - 0.25 * (eval_count / self.budget))  # Dynamic PAR
        self.bandwidth = max(0.05, self.bandwidth - 0.01)