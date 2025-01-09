import numpy as np

class HarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.HMCR = 0.7
        self.PAR = 0.3
        self.bw = 0.01
        
    def __call__(self, func):
        def initialize_harmony_memory():
            return np.random.uniform(self.lower_bound, self.upper_bound, (self.budget, self.dim))
        
        def mutate(x):
            for i in range(self.dim):
                if np.random.rand() < self.bw:
                    x[i] = np.clip(x[i] + np.random.normal(0, 1), self.lower_bound, self.upper_bound)
            return x
        
        def evaluate(x):
            return func(x)
        
        harmony_memory = initialize_harmony_memory()
        
        for _ in range(self.budget):
            new_harmony = np.array([mutate(h) if np.random.rand() < self.PAR else h for h in harmony_memory])
            new_harmony = np.array([h if np.random.rand() < self.HMCR else np.random.uniform(self.lower_bound, self.upper_bound, self.dim) for h in new_harmony])
            
            harmony_memory = np.where(np.array([evaluate(h) for h in new_harmony]) < np.array([evaluate(h) for h in harmony_memory]), new_harmony, harmony_memory)
        
        best_solution = harmony_memory[np.argmin([evaluate(h) for h in harmony_memory])]
        return best_solution