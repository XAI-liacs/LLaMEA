import numpy as np

class QuantumHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.harmony_memory_size = 30
        self.harmony_memory_consideration_rate = 0.95
        self.pitch_adjustment_rate = 0.7
        self.bw = 0.1
        self.history = []

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        harmony_memory_quantum = np.random.uniform(0, 1, (self.harmony_memory_size, self.dim))
        harmony_memory = lb + (ub - lb) * np.sin(np.pi * harmony_memory_quantum)
        fitness = np.array([func(x) for x in harmony_memory])
        best_idx = np.argmin(fitness)
        best_global = harmony_memory[best_idx]

        evaluations = self.harmony_memory_size

        while evaluations < self.budget:
            new_harmony = np.zeros(self.dim)
            for i in range(self.dim):
                if np.random.rand() < self.harmony_memory_consideration_rate:
                    new_harmony[i] = harmony_memory[np.random.randint(0, self.harmony_memory_size), i]
                    if np.random.rand() < self.pitch_adjustment_rate:
                        new_harmony[i] += self.bw * (np.random.rand() - 0.5) * (ub[i] - lb[i])
                else:
                    new_harmony[i] = lb[i] + np.random.rand() * (ub[i] - lb[i])

            new_harmony = np.clip(new_harmony, lb, ub)
            new_fitness = func(new_harmony)
            evaluations += 1

            if new_fitness < np.max(fitness):
                worst_idx = np.argmax(fitness)
                harmony_memory[worst_idx] = new_harmony
                fitness[worst_idx] = new_fitness
                if new_fitness < fitness[best_idx]:
                    best_idx = worst_idx
                    best_global = new_harmony

            self.history.append(best_global)

        return best_global