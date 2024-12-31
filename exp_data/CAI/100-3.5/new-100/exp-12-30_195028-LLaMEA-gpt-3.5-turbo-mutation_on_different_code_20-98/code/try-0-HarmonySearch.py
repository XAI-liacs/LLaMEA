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
    
    def generate_new_harmony(self, func, harmony_memory):
        new_harmony = np.zeros(self.dim)
        for i in range(self.dim):
            if np.random.rand() < self.pitch_adjust_rate:
                random_index = np.random.randint(0, len(harmony_memory))
                new_harmony[i] = harmony_memory[random_index][i] + np.random.uniform(-self.bandwidth, self.bandwidth)
            else:
                new_harmony[i] = np.random.uniform(func.bounds.lb, func.bounds.ub)
        return new_harmony
    
    def __call__(self, func):
        harmony_memory = [np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim) for _ in range(self.harmony_memory_size)]
        
        for _ in range(self.budget):
            new_harmony = self.generate_new_harmony(func, harmony_memory)
            f = func(new_harmony)
            
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = new_harmony
            
            worst_idx = np.argmax([func(h) for h in harmony_memory])
            if f < func(harmony_memory[worst_idx]):
                harmony_memory[worst_idx] = new_harmony
                
        return self.f_opt, self.x_opt