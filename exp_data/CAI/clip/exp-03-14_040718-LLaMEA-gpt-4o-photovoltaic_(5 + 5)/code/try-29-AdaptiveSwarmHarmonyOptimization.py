import numpy as np

class AdaptiveSwarmHarmonyOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 15 + 3 * dim
        self.harmony_memory_size = max(5, dim)
        self.harmony_memory = np.random.rand(self.harmony_memory_size, dim)
        self.phi = 0.5
        self.beta = 1.0
        self.gamma = 0.5
        self.diversity_threshold = 0.1  # Diversity threshold for adaptive control

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        self.harmony_memory = lb + (ub - lb) * np.random.rand(self.harmony_memory_size, self.dim)
        func_values = np.array([func(hm) for hm in self.harmony_memory])
        eval_count = self.harmony_memory_size

        while eval_count < self.budget:
            new_harmony = np.zeros(self.dim)
            for i in range(self.dim):
                if np.random.rand() < self.phi:
                    idx = np.random.randint(self.harmony_memory_size)
                    new_harmony[i] = self.harmony_memory[idx, i]
                else:
                    new_harmony[i] = lb[i] + (ub[i] - lb[i]) * np.random.rand()

                if np.random.rand() < self.gamma:
                    new_harmony[i] += self.beta * (ub[i] - lb[i]) * (np.random.rand() - 0.5)

            new_harmony = np.clip(new_harmony, lb, ub)
            new_value = func(new_harmony)
            eval_count += 1

            diversity = np.std(self.harmony_memory, axis=0).mean()
            if diversity < self.diversity_threshold:
                self.beta *= 1.1
                self.gamma *= 0.9
            else:
                self.beta *= 0.9
                self.gamma *= 1.1

            worst_idx = np.argmax(func_values)
            if new_value < func_values[worst_idx]:
                self.harmony_memory[worst_idx] = new_harmony
                func_values[worst_idx] = new_value

        best_idx = np.argmin(func_values)
        return self.harmony_memory[best_idx], func_values[best_idx]