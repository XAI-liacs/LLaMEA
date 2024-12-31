import numpy as np

class HarmonySearch:
    def __init__(self, budget=10000, dim=10, hm_size=20, bandwidth=0.01, pitch_adjust_rate=0.3):
        self.budget = budget
        self.dim = dim
        self.hm_size = hm_size
        self.bandwidth = bandwidth
        self.pitch_adjust_rate = pitch_adjust_rate
        self.f_opt = np.Inf
        self.x_opt = None

    def initialize_harmony_memory(self, func):
        self.harmony_memory = [np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim) for _ in range(self.hm_size)]

    def adjust_pitch(self, x):
        rand = np.random.rand(self.dim)
        x_new = x.copy()
        for i in range(self.dim):
            if rand[i] < self.pitch_adjust_rate:
                x_new[i] = x_new[i] + np.random.uniform(-self.bandwidth, self.bandwidth)
                x_new[i] = np.clip(x_new[i], func.bounds.lb, func.bounds.ub)
        return x_new

    def __call__(self, func):
        self.initialize_harmony_memory(func)
        
        for _ in range(self.budget):
            x = self.harmony_memory[np.random.randint(0, self.hm_size)]
            x_new = self.adjust_pitch(x)
            
            f = func(x_new)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x_new
                
                idx = np.argmax([func(x) for x in self.harmony_memory])
                if f < func(self.harmony_memory[idx]):
                    self.harmony_memory[idx] = x_new
                    
        return self.f_opt, self.x_opt