import numpy as np

class HarmonySearchImproved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.budget, self.dim))
        harmony_memory_fitness = np.array([func(harmony) for harmony in harmony_memory])
        
        for _ in range(self.budget):
            new_harmony = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
            new_harmony_fitness = func(new_harmony)
            
            if new_harmony_fitness < harmony_memory_fitness.max():
                index = harmony_memory_fitness.argmax()
                harmony_memory[index] = new_harmony
                harmony_memory_fitness[index] = new_harmony_fitness
                
            # Local search step
            for i in range(self.dim):
                delta = np.random.uniform(-0.05, 0.05)  # Small perturbation
                new_harmony_local = np.copy(new_harmony)
                new_harmony_local[i] += delta
                new_harmony_fitness_local = func(new_harmony_local)
                
                if new_harmony_fitness_local < new_harmony_fitness:
                    new_harmony = new_harmony_local
                    new_harmony_fitness = new_harmony_fitness_local
        
        best_harmony = harmony_memory[np.argmin(harmony_memory_fitness)]
        return best_harmony