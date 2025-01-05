import numpy as np

class Dynamic_Harmony_Search_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 30
        self.harmony_memory_consideration_rate = 0.95
        self.pitch_adjustment_rate = 0.7
        self.bandwidth = 0.10
        self.adaptive_threshold = 0.1
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        np.random.seed(42)

        harmony_memory = np.random.uniform(lb, ub, (self.harmony_memory_size, self.dim))
        harmony_values = np.array([func(h) for h in harmony_memory])
        best_harmony = harmony_memory[np.argmin(harmony_values)]
        best_value = np.min(harmony_values)
        
        evaluations = self.harmony_memory_size

        while evaluations < self.budget:
            new_harmony = np.empty(self.dim)
            for i in range(self.dim):
                if np.random.rand() < self.harmony_memory_consideration_rate:
                    new_harmony[i] = harmony_memory[np.random.randint(self.harmony_memory_size), i]
                    if np.random.rand() < self.pitch_adjustment_rate:
                        new_harmony[i] += np.random.uniform(-1, 1) * self.bandwidth
                else:
                    new_harmony[i] = np.random.uniform(lb[i], ub[i])
                    
            new_harmony = np.clip(new_harmony, lb, ub)
            new_value = func(new_harmony)
            evaluations += 1

            if new_value < best_value:
                best_harmony, best_value = new_harmony, new_value
            
            worst_index = np.argmax(harmony_values)
            if new_value < harmony_values[worst_index]:
                harmony_memory[worst_index] = new_harmony
                harmony_values[worst_index] = new_value

            # Adaptive Bandwidth and Threshold Adjustment
            self.bandwidth *= (1 - evaluations / self.budget)
            current_threshold = self.adaptive_threshold * (1 - evaluations / self.budget)
            if np.abs(harmony_values[worst_index] - new_value) < current_threshold:
                continue

        return best_harmony, best_value