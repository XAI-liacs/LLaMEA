import numpy as np

class EnhancedHarmonySearch:
    def __init__(self, budget, dim, harmony_memory_size=10, pitch_adjustment_rate=0.1, bandwidth=0.01):
        self.budget, self.dim, self.harmony_memory_size, self.pitch_adjustment_rate, self.bandwidth = budget, dim, harmony_memory_size, pitch_adjustment_rate, bandwidth

    def __call__(self, func):
        harmony_memory = np.random.uniform(-5.0, 5.0, (self.harmony_memory_size, self.dim))
        harmony_scores = np.array([func(h) for h in harmony_memory])
        
        for _ in range(self.budget):
            worst_index = np.argmax(harmony_scores)
            new_harmony = np.clip(harmony_memory[worst_index] + np.random.uniform(-self.bandwidth, self.bandwidth, (self.dim,)), -5.0, 5.0)
            new_score = func(new_harmony)
            if new_score < harmony_scores[worst_index]:
                harmony_memory[worst_index], harmony_scores[worst_index] = new_harmony, new_score
                
        return harmony_memory[np.argmin(harmony_scores)]