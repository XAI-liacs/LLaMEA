import numpy as np

class EnhancedAdaptiveHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 20  # Increased memory size for genetic diversity
        self.hmcr = 0.95  # Slightly increased HMCR for better convergence
        self.par = 0.4   # Increased PAR for more exploration
        self.bandwidth = 0.05  # Reduced bandwidth for fine-tuning

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        harmony_memory = np.random.uniform(lb, ub, (self.harmony_memory_size, self.dim))
        fitness = np.apply_along_axis(func, 1, harmony_memory)
        
        eval_count = self.harmony_memory_size
        best_idx = np.argmin(fitness)
        best_harmony = harmony_memory[best_idx].copy()
        best_fitness = fitness[best_idx]

        while eval_count < self.budget:
            if np.random.rand() < 0.5:  # Introduce a dual strategy with probability
                new_harmony = self._differential_evolution_step(harmony_memory, lb, ub, func)
            else:
                new_harmony = self._harmony_search_step(harmony_memory, lb, ub)

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

        return best_harmony, best_fitness

    def _harmony_search_step(self, harmony_memory, lb, ub):
        new_harmony = np.zeros(self.dim)
        for i in range(self.dim):
            if np.random.rand() < self.hmcr:
                contributor = np.random.randint(self.harmony_memory_size)
                new_harmony[i] = harmony_memory[contributor, i]
                if np.random.rand() < self.par:
                    new_harmony[i] += self.bandwidth * (np.random.rand() - 0.5) * (ub[i] - lb[i])
            else:
                new_harmony[i] = np.random.uniform(lb[i], ub[i])
        return new_harmony

    def _differential_evolution_step(self, harmony_memory, lb, ub, func):
        idxs = np.random.choice(self.harmony_memory_size, 3, replace=False)
        a, b, c = harmony_memory[idxs]
        F = np.random.uniform(0.6, 0.9)  # Dynamic differential weight
        trial = np.clip(a + F * (b - c), lb, ub)
        return trial

    def _adjust_parameters(self):
        self.hmcr = min(1.0, self.hmcr + 0.01) 
        self.par = max(0.2, self.par - 0.02)  # Adjusted more gradual reduction of PAR
        self.bandwidth = max(0.01, self.bandwidth * 0.995)