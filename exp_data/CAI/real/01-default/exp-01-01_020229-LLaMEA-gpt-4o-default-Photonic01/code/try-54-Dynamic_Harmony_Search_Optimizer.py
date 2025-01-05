import numpy as np

class Dynamic_Harmony_Search_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 30
        self.hmcr = 0.9  # Harmony Memory Consideration Rate
        self.par = 0.3  # Pitch Adjustment Rate
        self.bandwidth = 0.1
        self.layers = 3
        self.adaptive_rate = 0.95
        self.convergence_threshold = 1e-6

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        np.random.seed(42)

        # Initialize harmony memory
        harmony_memory = np.random.uniform(lb, ub, (self.harmony_memory_size, self.dim))
        harmony_values = np.array([func(h) for h in harmony_memory])
        best_harmony = harmony_memory[np.argmin(harmony_values)]
        best_value = np.min(harmony_values)
        
        evaluations = self.harmony_memory_size
        prev_best_value = best_value

        while evaluations < self.budget:
            new_harmony = np.empty(self.dim)
            for d in range(self.dim):
                if np.random.rand() < self.hmcr:
                    new_harmony[d] = harmony_memory[np.random.randint(self.harmony_memory_size), d]
                    if np.random.rand() < self.par:
                        new_harmony[d] += np.random.uniform(-1, 1) * self.bandwidth
                else:
                    new_harmony[d] = np.random.uniform(lb[d], ub[d])

            # Layer-based spatial exploitation
            for layer in range(self.layers):
                layer_factor = (self.layers - layer) / self.layers
                new_harmony += layer_factor * np.random.normal(0, self.bandwidth, self.dim)
                new_harmony = np.clip(new_harmony, lb, ub)
            
            current_value = func(new_harmony)
            evaluations += 1

            if current_value < np.max(harmony_values):
                worst_index = np.argmax(harmony_values)
                harmony_memory[worst_index] = new_harmony
                harmony_values[worst_index] = current_value

            if current_value < best_value:
                best_harmony = new_harmony
                best_value = current_value
            
            # Dynamic adaptation
            if prev_best_value - best_value < self.convergence_threshold:
                self.bandwidth *= self.adaptive_rate
            prev_best_value = best_value

            if evaluations >= self.budget:
                break

        return best_harmony, best_value