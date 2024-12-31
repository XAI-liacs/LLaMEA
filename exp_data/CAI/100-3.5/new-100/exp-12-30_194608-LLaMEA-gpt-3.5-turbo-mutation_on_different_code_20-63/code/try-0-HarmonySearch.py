import numpy as np

class HarmonySearch:
    def __init__(self, budget=10000, dim=10, harmony_memory_size=20, pitch_adjust_rate=0.3):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = harmony_memory_size
        self.pitch_adjust_rate = pitch_adjust_rate
        self.f_opt = np.Inf
        self.x_opt = None
        self.harmony_memory = np.random.uniform(-5.0, 5.0, (harmony_memory_size, dim))

    def __call__(self, func):
        for _ in range(self.budget):
            new_harmony = self.generate_new_harmony()
            f = func(new_harmony)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = new_harmony
        
        return self.f_opt, self.x_opt

    def generate_new_harmony(self):
        new_harmony = np.zeros(self.dim)
        
        for i in range(self.dim):
            if np.random.rand() < self.pitch_adjust_rate:
                new_harmony[i] = np.random.uniform(-5.0, 5.0)
            else:
                idx = np.random.randint(self.harmony_memory_size)
                new_harmony[i] = self.harmony_memory[idx, i]
        
        return new_harmony