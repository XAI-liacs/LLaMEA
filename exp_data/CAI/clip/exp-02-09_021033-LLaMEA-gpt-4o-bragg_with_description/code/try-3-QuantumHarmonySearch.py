import numpy as np

class QuantumHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 10 * dim
        self.harmony_memory = None
        self.harmony_memory_consideration_rate = 0.95
        self.pitch_adjustment_rate = 0.5
        self.bounds = None
    
    def __call__(self, func):
        self.bounds = (func.bounds.lb, func.bounds.ub)
        self.harmony_memory = np.random.uniform(self.bounds[0], self.bounds[1], (self.harmony_memory_size, self.dim))
        fitness = np.apply_along_axis(func, 1, self.harmony_memory)
        evaluations = self.harmony_memory_size
        best_idx = np.argmin(fitness)
        best_solution = self.harmony_memory[best_idx]
        best_fitness = fitness[best_idx]
        
        while evaluations < self.budget:
            new_harmony = np.zeros(self.dim)

            for i in range(self.dim):
                if np.random.rand() < self.harmony_memory_consideration_rate:
                    harmony_index = np.random.randint(self.harmony_memory_size)
                    new_harmony[i] = self.harmony_memory[harmony_index, i]
                    if np.random.rand() < self.pitch_adjustment_rate:
                        adaptive_step = (self.bounds[1] - self.bounds[0]) * (1 - evaluations / self.budget) * np.random.normal()
                        new_harmony[i] += adaptive_step
                        new_harmony[i] = np.clip(new_harmony[i], self.bounds[0], self.bounds[1])
                else:
                    new_harmony[i] = np.random.uniform(self.bounds[0], self.bounds[1])

            new_fitness = func(new_harmony)
            evaluations += 1

            if new_fitness < best_fitness:
                best_solution = new_harmony
                best_fitness = new_fitness

            worst_idx = np.argmax(fitness)
            if new_fitness < fitness[worst_idx]:
                self.harmony_memory[worst_idx] = new_harmony
                fitness[worst_idx] = new_fitness
                
        return best_solution