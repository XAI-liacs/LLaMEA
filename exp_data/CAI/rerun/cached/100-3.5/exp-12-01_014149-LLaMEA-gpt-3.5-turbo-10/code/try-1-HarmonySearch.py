import numpy as np

class HarmonySearch:
    def __init__(self, budget, dim, harmony_memory_size=20, pitch_adjust_rate=0.5, bandwidth=0.01):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = harmony_memory_size
        self.pitch_adjust_rate = pitch_adjust_rate
        self.bandwidth = bandwidth

    def generate_harmony(self):
        return np.random.uniform(-5.0, 5.0, self.dim)

    def __call__(self, func):
        harmony_memory = [self.generate_harmony() for _ in range(self.harmony_memory_size)]
        for _ in range(self.budget):
            new_harmony = np.clip(np.random.normal(np.mean(harmony_memory, axis=0), self.bandwidth), -5.0, 5.0)
            if func(new_harmony) < func(harmony_memory[-1]):
                harmony_memory[-1] = new_harmony
                harmony_memory = sorted(harmony_memory, key=lambda x: func(x))
        return harmony_memory[0]