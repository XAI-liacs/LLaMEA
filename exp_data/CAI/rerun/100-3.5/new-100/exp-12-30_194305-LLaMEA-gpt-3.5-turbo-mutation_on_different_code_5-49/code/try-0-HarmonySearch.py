import numpy as np

class HarmonySearch:
    def __init__(self, budget=10000, dim=10, harmony_memory_size=20, bandwidth=0.01):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = harmony_memory_size
        self.bandwidth = bandwidth
        self.f_opt = np.Inf
        self.x_opt = None

    def generate_initial_harmony_memory(self, func):
        harmony_memory = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.harmony_memory_size, self.dim))
        return harmony_memory

    def improvisation(self, harmony_memory, func):
        new_harmony = np.copy(harmony_memory[np.random.randint(0, self.harmony_memory_size)])
        idx = np.random.choice(self.dim)
        new_harmony[idx] = np.random.uniform(new_harmony[idx] - self.bandwidth, new_harmony[idx] + self.bandwidth)
        return new_harmony

    def __call__(self, func):
        harmony_memory = self.generate_initial_harmony_memory(func)
        
        for i in range(self.budget):
            new_harmony = self.improvisation(harmony_memory, func)
            f = func(new_harmony)
            
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = new_harmony
                
                # Update harmony memory
                worst_idx = np.argmax(func(harmony_memory))
                harmony_memory[worst_idx] = new_harmony
                
        return self.f_opt, self.x_opt