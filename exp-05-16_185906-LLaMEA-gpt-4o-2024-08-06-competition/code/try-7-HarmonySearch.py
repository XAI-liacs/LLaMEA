import numpy as np

class HarmonySearch:
    def __init__(self, budget=10000, dim=10, harmony_memory_size=5, pitch_adjust_rate=0.3, bandwidth=1.0):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = harmony_memory_size
        self.pitch_adjust_rate = pitch_adjust_rate
        self.bandwidth = bandwidth
        self.harmonies = np.random.uniform(-100, 100, (self.harmony_memory_size, self.dim))
        self.harmony_memory = np.array([np.Inf] * self.harmony_memory_size)
    
    def __call__(self, func):
        evaluations = 0
        for i in range(self.harmony_memory_size):
            self.harmony_memory[i] = func(self.harmonies[i])
            evaluations += 1
        
        while evaluations < self.budget:
            new_harmony = np.zeros(self.dim)
            for j in range(self.dim):
                if np.random.rand() < self.pitch_adjust_rate * (1 - evaluations/self.budget):
                    idx = np.random.randint(0, self.harmony_memory_size)
                    new_harmony[j] = self.harmonies[idx][j] + np.random.uniform(-self.bandwidth, self.bandwidth)
                else:
                    new_harmony[j] = np.random.uniform(-100, 100)
            
            new_harmony = np.clip(new_harmony, -100, 100)
            new_value = func(new_harmony)
            evaluations += 1
            
            if new_value < max(self.harmony_memory):
                max_index = np.argmax(self.harmony_memory)
                self.harmony_memory[max_index] = new_value
                self.harmonies[max_index] = new_harmony
                self.bandwidth *= 0.99  # Adjust bandwidth dynamically
        
        best_index = np.argmin(self.harmony_memory)
        return self.harmony_memory[best_index], self.harmonies[best_index]