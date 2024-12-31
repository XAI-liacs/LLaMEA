import numpy as np

class HarmonySearch:
    def __init__(self, budget=10000, dim=10, harmony_memory_size=10, pitch_adjust_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = harmony_memory_size
        self.pitch_adjust_rate = pitch_adjust_rate
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        harmony_memory = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.harmony_memory_size, self.dim))

        for _ in range(self.budget):
            new_harmony = np.zeros(self.dim)
            for d in range(self.dim):
                if np.random.rand() < self.pitch_adjust_rate:
                    new_harmony[d] = np.random.uniform(func.bounds.lb, func.bounds.ub)
                else:
                    idx = np.random.randint(0, self.harmony_memory_size)
                    new_harmony[d] = harmony_memory[idx, d]

            f = func(new_harmony)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = new_harmony

            idx = np.argmax(harmony_memory, axis=0)
            idx_to_replace = np.argmax(func(harmony_memory), axis=0)
            harmony_memory[idx_to_replace] = harmony_memory[idx]

        return self.f_opt, self.x_opt