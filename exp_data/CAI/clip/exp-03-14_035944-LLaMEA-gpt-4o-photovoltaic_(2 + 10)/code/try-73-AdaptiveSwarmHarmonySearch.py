import numpy as np

class AdaptiveSwarmHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        memory_size = 10
        harmony_memory = np.random.uniform(lb, ub, (memory_size, self.dim))
        fitness_memory = np.array([func(harmony) for harmony in harmony_memory])
        best_harmony = harmony_memory[np.argmin(fitness_memory)]
        best_fitness = np.min(fitness_memory)
        eval_count = memory_size

        while eval_count < self.budget:
            learning_factor = 1.0 - eval_count / self.budget
            new_harmony = self.generate_harmony(harmony_memory, lb, ub, best_harmony, learning_factor)
            new_fitness = func(new_harmony)
            eval_count += 1

            worst_idx = np.argmax(fitness_memory)
            if new_fitness < fitness_memory[worst_idx]:
                harmony_memory[worst_idx] = new_harmony
                fitness_memory[worst_idx] = new_fitness

            if new_fitness < best_fitness:
                best_harmony = new_harmony
                best_fitness = new_fitness

            if eval_count % (self.budget // 10) == 0:
                self.adaptive_adjustment(harmony_memory, fitness_memory)

        return best_harmony

    def generate_harmony(self, memory, lb, ub, best_harmony, learning_factor):
        harmony = np.empty(self.dim)
        for i in range(self.dim):
            if np.random.rand() < 0.9:  # Increased probability for memory selection
                harmony[i] = memory[np.random.randint(len(memory)), i]
            else:
                harmony[i] = np.random.uniform(lb[i], ub[i])

            if np.random.rand() < 0.1:
                adjustment = (ub[i] - lb[i]) * (np.random.rand() - 0.5)
                harmony[i] = np.clip(harmony[i] + adjustment, lb[i], ub[i])

            harmony[i] += learning_factor * 0.05 * (best_harmony[i] - harmony[i])
        
        return harmony

    def adaptive_adjustment(self, memory, fitness_memory):
        variation = np.std(memory, axis=0)
        if np.std(fitness_memory) > 0.1:
            for harmony in memory:
                harmony += np.random.normal(0, 0.1, size=self.dim) * variation