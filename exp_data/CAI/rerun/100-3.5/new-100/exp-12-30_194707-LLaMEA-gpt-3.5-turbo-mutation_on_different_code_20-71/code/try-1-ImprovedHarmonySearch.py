import numpy as np

class ImprovedHarmonySearch:
    def __init__(self, budget=10000, dim=10, harmony_memory_size=20, pitch_adjustment_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = harmony_memory_size
        self.pitch_adjustment_rate = pitch_adjustment_rate
        self.f_opt = np.Inf
        self.x_opt = None
        self.harmony_memory = np.random.uniform(-5.0, 5.0, (self.harmony_memory_size, self.dim))

    def __call__(self, func):
        for i in range(self.budget):
            if np.random.rand() < self.pitch_adjustment_rate:
                x = np.random.uniform(-5.0, 5.0, self.dim)
            else:
                x = self.harmony_memory[np.random.randint(self.harmony_memory_size)]

            f = func(x)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x
                idx = np.argmax(self.harmony_memory_fitness) if 'harmony_memory_fitness' in locals() else np.random.randint(self.harmony_memory_size)
                self.harmony_memory[idx] = x
                if 'harmony_memory_fitness' in locals():
                    self.harmony_memory_fitness[idx] = f

            self.pitch_adjustment_rate = 0.1 + 0.9 * (1 - i / self.budget)  # Adaptive pitch adjustment rate

        return self.f_opt, self.x_opt