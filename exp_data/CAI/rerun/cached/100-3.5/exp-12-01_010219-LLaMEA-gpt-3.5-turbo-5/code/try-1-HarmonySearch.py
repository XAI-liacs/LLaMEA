import numpy as np

class HarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        
    def __call__(self, func):
        harmony_memory_size = 20
        bandwidth = 0.01
        pitch_adjust_rate = 0.5
        
        def initialize_harmony_memory():
            return np.random.uniform(self.lower_bound, self.upper_bound, (harmony_memory_size, self.dim))
        
        def improvise(harmony_memory):
            new_harmony = np.copy(harmony_memory)
            for i in range(self.dim):
                rand = np.random.rand()
                if rand < pitch_adjust_rate:
                    rand_index = np.random.randint(harmony_memory_size)
                    new_harmony[rand_index, i] = np.clip(harmony_memory[rand_index, i] + bandwidth * np.random.randn(), self.lower_bound, self.upper_bound)
            return new_harmony
        
        harmony_memory = initialize_harmony_memory()
        
        for _ in range(self.budget):
            new_solution = improvise(harmony_memory)
            new_fitness = func(new_solution)
            worst_index = np.argmax(new_fitness)
            if new_fitness[worst_index] < func(harmony_memory[np.argmin(new_fitness)]):
                harmony_memory[worst_index] = new_solution[worst_index]
        
        best_index = np.argmin(func(harmony_memory))
        return harmony_memory[best_index]