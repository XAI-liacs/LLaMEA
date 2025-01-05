import numpy as np

class Quantum_Harmony_Search_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 20
        self.harmony_consideration_rate = 0.9
        self.pitch_adjustment_rate = 0.3
        self.noise_scale = 0.1
        self.adaptive_rate = 0.98

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        np.random.seed(42)

        harmony_memory = np.random.uniform(lb, ub, (self.harmony_memory_size, self.dim))
        harmony_values = np.array([func(harmony) for harmony in harmony_memory])
        best_harmony = harmony_memory[np.argmin(harmony_values)]
        best_value = np.min(harmony_values)
        
        evaluations = self.harmony_memory_size

        while evaluations < self.budget:
            new_harmony = np.zeros(self.dim)
            for j in range(self.dim):
                if np.random.rand() < self.harmony_consideration_rate:
                    # Harmony memory consideration
                    idx = np.random.randint(self.harmony_memory_size)
                    new_harmony[j] = harmony_memory[idx, j]
                    
                    # Pitch adjustment
                    if np.random.rand() < self.pitch_adjustment_rate:
                        pitch_adjustment = np.random.uniform(-self.noise_scale, self.noise_scale)
                        new_harmony[j] += pitch_adjustment
                else:
                    # Random selection
                    new_harmony[j] = np.random.uniform(lb[j], ub[j])

            new_harmony = np.clip(new_harmony, lb, ub)
            current_value = func(new_harmony)
            evaluations += 1

            if current_value < best_value:
                best_harmony = new_harmony
                best_value = current_value

            # Update harmony memory if the new harmony is better
            worst_idx = np.argmax(harmony_values)
            if current_value < harmony_values[worst_idx]:
                harmony_memory[worst_idx] = new_harmony
                harmony_values[worst_idx] = current_value

            # Adaptive noise scale adjustment
            self.noise_scale *= self.adaptive_rate

        return best_harmony, best_value