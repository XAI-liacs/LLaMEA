import numpy as np

class EnhancedQuantumHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 12
        self.harmony_memory = np.random.rand(self.harmony_memory_size, self.dim)
        self.harmony_memory_improv_rate = 0.95
        self.pitch_adjust_rate = 0.35
        self.bandwidth = 0.05

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.harmony_memory = lb + (ub - lb) * self.harmony_memory

        best_solution = None
        best_fitness = float('inf')

        for eval_count in range(self.budget):
            new_harmony = np.zeros(self.dim)
            dynamic_improv_rate = self.harmony_memory_improv_rate * (1 - eval_count / self.budget)
            for i in range(self.dim):
                if np.random.rand() < dynamic_improv_rate:
                    idx = np.random.choice(self.harmony_memory_size)
                    new_harmony[i] = self.harmony_memory[idx, i]
                    if np.random.rand() < self.pitch_adjust_rate:
                        dynamic_bandwidth = self.bandwidth * (1 - eval_count / self.budget)
                        new_harmony[i] += dynamic_bandwidth * np.random.uniform(-1, 1)
                        new_harmony[i] = np.clip(new_harmony[i], lb[i], ub[i])
                else:
                    new_harmony[i] = lb[i] + (ub[i] - lb[i]) * np.random.rand()

            fitness = func(new_harmony)
            if fitness < best_fitness:
                best_fitness = fitness
                best_solution = new_harmony

            worst_idx = np.argmax([func(h) for h in self.harmony_memory])
            if fitness < func(self.harmony_memory[worst_idx]):
                self.harmony_memory[worst_idx] = new_harmony

        return best_solution