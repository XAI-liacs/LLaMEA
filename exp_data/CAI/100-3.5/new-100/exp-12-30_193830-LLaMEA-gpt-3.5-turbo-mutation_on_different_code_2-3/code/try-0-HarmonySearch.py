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
        
        for _ in range(self.budget):
            if np.random.rand() < self.pitch_adjust_rate:
                new_harmony = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
            else:
                idx = np.random.randint(self.harmony_memory_size)
                new_harmony = np.copy(harmony_memory[idx])
                mask = np.random.rand(self.dim) < 0.3
                new_harmony[mask] = np.random.uniform(func.bounds.lb, func.bounds.ub, size=np.sum(mask))
            
            f = func(new_harmony)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = new_harmony
            
            worst_idx = np.argmax(f)
            if f[worst_idx] < self.f_opt:
                harmony_memory[worst_idx] = new_harmony
        
        return self.f_opt, self.x_opt