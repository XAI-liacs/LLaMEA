import numpy as np

class Stochastic_Quantum_Harmony_Search_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 30
        self.harmony_consideration_rate = 0.95
        self.pitch_adjustment_rate = 0.5
        self.q_factor = 0.5
        self.memory_rate_decay = 0.995
        self.scale_adjustment = 0.2
        self.adaptive_memory_rate = 0.05

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        np.random.seed(42)

        # Initialize harmony memory randomly
        harmony_memory = np.random.uniform(lb, ub, (self.harmony_memory_size, self.dim))
        harmony_values = np.array([func(hm) for hm in harmony_memory])
        best_harmony = harmony_memory[np.argmin(harmony_values)]
        best_value = np.min(harmony_values)
        
        evaluations = self.harmony_memory_size
        
        while evaluations < self.budget:
            new_harmony = np.copy(best_harmony)

            for i in range(self.dim):
                if np.random.rand() < self.harmony_consideration_rate:
                    # Select from harmony memory
                    new_harmony[i] = harmony_memory[np.random.randint(self.harmony_memory_size)][i]
                if np.random.rand() < self.pitch_adjustment_rate:
                    # Adjust pitch with quantum-inspired randomization
                    new_harmony[i] += self.q_factor * np.random.normal(scale=self.scale_adjustment)
                    new_harmony[i] = np.clip(new_harmony[i], lb[i], ub[i])

            current_value = func(new_harmony)
            evaluations += 1

            # Replace worst harmony if new harmony is better
            if current_value < np.max(harmony_values):
                worst_index = np.argmax(harmony_values)
                harmony_memory[worst_index] = new_harmony
                harmony_values[worst_index] = current_value

            # Update best harmony found
            if current_value < best_value:
                best_harmony = new_harmony
                best_value = current_value

            # Adaptive adjustment of parameters
            self.harmony_consideration_rate *= self.memory_rate_decay
            self.q_factor += self.adaptive_memory_rate * (1 - evaluations / self.budget)

        return best_harmony, best_value