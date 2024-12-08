import numpy as np

class EnhancedHarmonySearch:
    def __init__(self, budget, dim, harmony_memory_size=10, pitch_adjustment_rate=0.1, bandwidth=0.01):
        self.budget, self.dim, self.harmony_memory_size, self.pitch_adjustment_rate, self.bandwidth = budget, dim, harmony_memory_size, pitch_adjustment_rate, bandwidth

    def __call__(self, func):
        def initialize_harmony_memory():
            return np.random.uniform(-5.0, 5.0, (self.harmony_memory_size, self.dim))

        def generate_new_harmony(harmony_memory):
            random_index = np.random.randint(self.harmony_memory_size)
            new_harmony = np.copy(harmony_memory[random_index])
            for i in range(self.dim):
                if np.random.rand() < self.pitch_adjustment_rate:
                    new_harmony[i] += np.random.uniform(-self.bandwidth, self.bandwidth)
                    new_harmony[i] = np.clip(new_harmony[i], -5.0, 5.0)
            return new_harmony

        def evaluate_harmony(harmony):
            return func(harmony)

        harmony_memory = initialize_harmony_memory()
        harmony_scores = np.array([evaluate_harmony(h) for h in harmony_memory])
        for _ in range(self.budget):
            new_harmony = generate_new_harmony(harmony_memory)
            new_score = evaluate_harmony(new_harmony)
            min_index = np.argmin(harmony_scores)
            if new_score < harmony_scores[min_index]:
                harmony_memory[min_index], harmony_scores[min_index] = new_harmony, new_score
        return harmony_memory[np.argmin(harmony_scores)]