import numpy as np

class HarmonySearch:
    def __init__(self, budget, dim, harmony_memory_size=10, pitch_adjust_rate=0.5):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = harmony_memory_size
        self.pitch_adjust_rate = pitch_adjust_rate

    def __call__(self, func):
        def initialize_harmony_memory():
            return np.random.uniform(-5.0, 5.0, (self.harmony_memory_size, self.dim))

        def generate_new_harmony(memory):
            new_harmony = []
            for i in range(self.dim):
                if np.random.rand() < self.pitch_adjust_rate:
                    new_harmony.append(np.random.uniform(-5.0, 5.0))
                else:
                    idx = np.random.randint(self.harmony_memory_size)
                    new_harmony.append(memory[idx, i])
            return np.array(new_harmony)

        def evaluate_harmony(harmony):
            return func(harmony)

        harmony_memory = initialize_harmony_memory()
        for _ in range(self.budget):
            new_harmony = generate_new_harmony(harmony_memory)
            if evaluate_harmony(new_harmony) < evaluate_harmony(harmony_memory[-1]):
                harmony_memory[-1] = new_harmony
                harmony_memory = harmony_memory[np.argsort([evaluate_harmony(h) for h in harmony_memory])]

        best_solution = min(harmony_memory, key=evaluate_harmony)
        return best_solution