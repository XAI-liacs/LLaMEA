import numpy as np

class AdaptiveHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        
    def __call__(self, func):
        def initialize_harmony(memory_size, dim):
            return np.random.uniform(-5.0, 5.0, (memory_size, dim))
        
        def evaluate_harmony(harmony, func):
            return np.array([func(h) for h in harmony])
        
        memory_size = 10
        harmony_memory = initialize_harmony(memory_size, self.dim)
        harmony_fitness = evaluate_harmony(harmony_memory, func)
        
        for _ in range(self.budget - memory_size):
            step_size = np.random.uniform(0.1, 1.0)
            new_harmony = np.clip(harmony_memory[np.argsort(harmony_fitness)[:2]] + np.random.normal(0, step_size, (2, self.dim)), -5.0, 5.0)
            new_fitness = evaluate_harmony(new_harmony, func)
            replace_idx = np.argmax(harmony_fitness)
            if new_fitness.min() < harmony_fitness[replace_idx]:
                harmony_memory[replace_idx] = new_harmony[new_fitness.argmin()]
                harmony_fitness[replace_idx] = new_fitness.min()
        
        return harmony_memory[harmony_fitness.argmin()]