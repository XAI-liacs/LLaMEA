import numpy as np

class HarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.hmcr = 0.7
        self.par = 0.4
        self.bandwidth = 0.01
        
    def generate_initial_harmony(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
    
    def pitch_adjustment(self, harmony):
        new_harmony = np.copy(harmony)
        for i in range(len(new_harmony)):
            if np.random.rand() < self.hmcr:
                if np.random.rand() < self.par:
                    new_harmony[i] = new_harmony[i] + np.random.uniform(-self.bandwidth, self.bandwidth)
        return np.clip(new_harmony, self.lower_bound, self.upper_bound)
    
    def __call__(self, func):
        harmony_memory = [self.generate_initial_harmony() for _ in range(10)]
        harmony_memory_fit = [func(harmony) for harmony in harmony_memory]
        
        for _ in range(self.budget - 10):
            new_harmony = self.pitch_adjustment(harmony_memory[np.argmin(harmony_memory_fit)])
            new_fit = func(new_harmony)
            if new_fit < max(harmony_memory_fit):
                idx = np.argmax(harmony_memory_fit)
                harmony_memory[idx] = new_harmony
                harmony_memory_fit[idx] = new_fit
        
        return harmony_memory[np.argmin(harmony_memory_fit)]