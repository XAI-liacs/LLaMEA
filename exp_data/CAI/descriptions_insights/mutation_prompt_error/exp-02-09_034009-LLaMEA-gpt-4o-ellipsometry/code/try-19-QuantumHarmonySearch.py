import numpy as np

class QuantumHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 10
        self.harmony_memory = np.random.rand(self.harmony_memory_size, self.dim)
        self.harmony_memory *= (1 + 1) - 1
        self.harmony_memory_improv_rate = 0.9
        self.pitch_adjust_rate = 0.4  # Adjusted value
        self.bandwidth = 0.05

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.harmony_memory = lb + (ub - lb) * self.harmony_memory

        best_solution = None
        best_fitness = float('inf')

        for _ in range(self.budget):
            # Adjust harmony memory size dynamically
            self.harmony_memory_size = max(5, int(0.05 * self.budget))  # Dynamic adjustment
            new_harmony = np.zeros(self.dim)
            for i in range(self.dim):
                if np.random.rand() < self.harmony_memory_improv_rate * (1 - self.pitch_adjust_rate):
                    idx = np.random.choice(np.arange(self.harmony_memory_size), p=np.ones(self.harmony_memory_size)/self.harmony_memory_size)
                    new_harmony[i] = self.harmony_memory[idx, i]
                    if np.random.rand() < self.pitch_adjust_rate:
                        self.bandwidth = 0.02 + 0.08 * np.random.rand()  # Adjusted value
                        new_harmony[i] += self.bandwidth * np.random.uniform(-1, 1)
                        new_harmony[i] = np.clip(new_harmony[i], lb[i], ub[i])
                else:
                    new_harmony[i] = lb[i] + (ub[i] - lb[i]) * np.random.rand()

            fitness = func(new_harmony)
            if fitness < best_fitness:
                best_fitness = fitness
                best_solution = new_harmony

            worst_idx = np.argmax([func(h) for h in self.harmony_memory])
            if func(new_harmony) < func(self.harmony_memory[worst_idx]):
                self.harmony_memory[worst_idx] = new_harmony

        return best_solution