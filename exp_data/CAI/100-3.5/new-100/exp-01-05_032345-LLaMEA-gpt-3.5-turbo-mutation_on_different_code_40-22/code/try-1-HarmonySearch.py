import numpy as np

class HarmonySearch:
    def __init__(self, budget=10000, dim=10, harmony_memory_size=10, bandwidth=0.01, pitch_adjust_rate=0.5):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = harmony_memory_size
        self.bandwidth = bandwidth
        self.pitch_adjust_rate = pitch_adjust_rate
        self.f_opt = np.Inf
        self.x_opt = None
        self.harmony_memory = np.random.uniform(-5.0, 5.0, size=(harmony_memory_size, dim))

    def __call__(self, func):
        for i in range(self.budget):
            new_harmony = np.mean(self.harmony_memory, axis=0) + np.random.uniform(-self.bandwidth, self.bandwidth, size=self.dim)
            new_harmony = np.clip(new_harmony, -5.0, 5.0)
            
            if np.random.rand() < self.pitch_adjust_rate:
                pitch_adjusted_dim = np.random.randint(self.dim)
                new_harmony[pitch_adjusted_dim] = np.random.uniform(-5.0, 5.0)
            
            f = func(new_harmony)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = new_harmony
                worst_index = np.argmax([func(h) for h in self.harmony_memory])
                self.harmony_memory[worst_index] = new_harmony
            
        return self.f_opt, self.x_opt