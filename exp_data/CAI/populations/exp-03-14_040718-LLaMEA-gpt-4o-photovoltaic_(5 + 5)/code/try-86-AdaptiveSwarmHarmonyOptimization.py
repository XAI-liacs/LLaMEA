import numpy as np

class AdaptiveSwarmHarmonyOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + 2 * dim
        self.harmony_memory_size = max(5, dim)
        self.harmony_memory = np.random.rand(self.harmony_memory_size, dim)
        self.secondary_memory_size = max(3, dim // 2)
        self.secondary_memory = np.random.rand(self.secondary_memory_size, dim)
        self.phi = 0.5
        self.beta = 1.0
        self.gamma = 0.5
        self.dynamic_adaptation_rate = 0.005

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        self.harmony_memory = lb + (ub - lb) * np.random.rand(self.harmony_memory_size, self.dim)
        self.secondary_memory = lb + (ub - lb) * np.random.rand(self.secondary_memory_size, self.dim)
        func_values = np.array([func(hm) for hm in self.harmony_memory])
        secondary_values = np.array([func(sm) for sm in self.secondary_memory])
        eval_count = self.harmony_memory_size + self.secondary_memory_size

        while eval_count < self.budget:
            new_harmony = np.zeros(self.dim)
            for i in range(self.dim):
                if np.random.rand() < self.phi:
                    if np.random.rand() < 0.5:
                        idx = np.random.randint(self.harmony_memory_size)
                        new_harmony[i] = self.harmony_memory[idx, i]
                    else:
                        idx = np.random.randint(self.secondary_memory_size)
                        new_harmony[i] = self.secondary_memory[idx, i]
                else:
                    new_harmony[i] = lb[i] + (ub[i] - lb[i]) * np.random.rand()
                if np.random.rand() < self.gamma:
                    self.beta = np.random.uniform(0.5, 1.5)
                    # Change made here: adaptive beta scaling
                    new_harmony[i] += self.beta * (ub[i] - lb[i]) * (np.random.rand() - 0.5) * (1 - eval_count / self.budget)

            new_harmony = np.clip(new_harmony, lb, ub)
            new_value = func(new_harmony)
            eval_count += 1

            if new_value < max(func_values.max(), secondary_values.max()):
                if new_value < func_values.max():
                    worst_idx = np.argmax(func_values)
                    self.harmony_memory[worst_idx] = new_harmony
                    func_values[worst_idx] = new_value
                else:
                    worst_idx = np.argmax(secondary_values)
                    self.secondary_memory[worst_idx] = new_harmony
                    secondary_values[worst_idx] = new_value

                self.phi = min(1.0, self.phi + self.dynamic_adaptation_rate)
                self.harmony_memory_size = min(self.population_size, int(self.harmony_memory_size * 1.05))  # Adaptive memory size

            self.gamma = max(0.1, self.gamma * 0.97)  # Self-tuning gamma

        best_idx = np.argmin(func_values)
        return self.harmony_memory[best_idx], func_values[best_idx]