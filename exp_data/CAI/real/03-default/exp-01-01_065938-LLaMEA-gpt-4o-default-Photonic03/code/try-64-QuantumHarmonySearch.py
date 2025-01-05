import numpy as np

class QuantumHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 50
        self.harmony_consideration_rate = 0.8
        self.pitch_adjustment_rate = 0.3
        self.scale = 0.2

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        harmony_memory_quantum = np.random.uniform(0, 1, (self.harmony_memory_size, self.dim))
        harmony_memory = lb + (ub - lb) * np.sin(np.pi * harmony_memory_quantum)
        fitness = np.array([func(x) for x in harmony_memory])
        best_idx = np.argmin(fitness)
        best_solution = harmony_memory[best_idx]

        evaluations = self.harmony_memory_size

        while evaluations < self.budget:
            new_harmony = np.zeros(self.dim)

            for i in range(self.dim):
                if np.random.rand() < self.harmony_consideration_rate:
                    new_harmony[i] = harmony_memory[np.random.randint(self.harmony_memory_size)][i]
                    if np.random.rand() < self.pitch_adjustment_rate:
                        new_harmony[i] += self.scale * (np.random.rand() - 0.5)
                else:
                    new_harmony[i] = lb[i] + (ub[i] - lb[i]) * np.random.rand()

            new_harmony = np.clip(new_harmony, lb, ub)
            new_fitness = func(new_harmony)
            evaluations += 1

            if new_fitness < fitness[best_idx]:
                best_idx = np.argmax(fitness)
                harmony_memory[best_idx] = new_harmony
                fitness[best_idx] = new_fitness
                if new_fitness < fitness[best_idx]:
                    best_solution = new_harmony

            self.harmony_consideration_rate = np.clip(self.harmony_consideration_rate + 0.01 * (np.random.rand() - 0.5), 0.7, 0.9)
            self.pitch_adjustment_rate = np.clip(self.pitch_adjustment_rate + 0.01 * (np.random.rand() - 0.5), 0.2, 0.4)

        return best_solution