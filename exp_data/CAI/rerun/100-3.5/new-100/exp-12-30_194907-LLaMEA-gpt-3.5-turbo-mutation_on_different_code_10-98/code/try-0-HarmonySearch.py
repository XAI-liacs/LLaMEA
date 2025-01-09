import numpy as np

class HarmonySearch:
    def __init__(self, budget=10000, dim=10, harmony_memory_size=20, pitch_adjust_rate=0.5):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = harmony_memory_size
        self.pitch_adjust_rate = pitch_adjust_rate
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        harmony_memory = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.harmony_memory_size, self.dim))
        harmony_memory_fitness = np.array([func(x) for x in harmony_memory])

        for i in range(self.budget):
            if np.random.rand() < self.pitch_adjust_rate:
                x = np.random.choice(harmony_memory)
            else:
                x = np.mean(harmony_memory, axis=0)
                
            f = func(x)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x
                idx = np.argmax(harmony_memory_fitness)
                harmony_memory[idx] = x
                harmony_memory_fitness[idx] = f

        return self.f_opt, self.x_opt