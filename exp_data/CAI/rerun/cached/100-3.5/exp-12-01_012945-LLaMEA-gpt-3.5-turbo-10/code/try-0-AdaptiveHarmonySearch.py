import numpy as np

class AdaptiveHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 20
        self.bandwidth = 0.01
        self.HMCR = 0.9
        self.PAR = 0.4
        self.current_evals = 0
        self.harmony_memory = np.random.uniform(-5.0, 5.0, (self.harmony_memory_size, self.dim))

    def __call__(self, func):
        while self.current_evals < self.budget:
            new_harmony = np.zeros(self.dim)
            for i in range(self.dim):
                if np.random.rand() < self.HMCR:
                    if np.random.rand() < self.PAR:
                        new_harmony[i] = self.harmony_memory[np.random.randint(self.harmony_memory_size), i]
                    else:
                        new_harmony[i] = np.random.uniform(-5.0, 5.0)
                else:
                    new_harmony[i] = self.harmony_memory[np.random.randint(self.harmony_memory_size), i] + np.random.uniform(-self.bandwidth, self.bandwidth)
                    
            if func(new_harmony) < func(self.harmony_memory[np.argmin(func(self.harmony_memory)), :]):
                self.harmony_memory[np.argmax(func(self.harmony_memory)), :] = new_harmony
            self.current_evals += 1
        
        return self.harmony_memory[np.argmin(func(self.harmony_memory)), :]