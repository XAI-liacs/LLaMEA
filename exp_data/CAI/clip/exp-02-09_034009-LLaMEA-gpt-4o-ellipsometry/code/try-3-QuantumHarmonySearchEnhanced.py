import numpy as np

class QuantumHarmonySearchEnhanced:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 10
        self.harmony_memory = np.random.rand(self.harmony_memory_size, self.dim)
        self.harmony_memory *= (1 + 1) - 1
        self.harmony_memory_improv_rate = 0.9
        self.pitch_adjust_rate = 0.3
        self.bandwidth = 0.05
        self.best_solutions_history = []

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.harmony_memory = lb + (ub - lb) * self.harmony_memory

        best_solution = None
        best_fitness = float('inf')

        for eval_count in range(self.budget):
            new_harmony = np.zeros(self.dim)
            for i in range(self.dim):
                if np.random.rand() < self.harmony_memory_improv_rate:
                    idx = np.random.randint(self.harmony_memory_size)
                    new_harmony[i] = self.harmony_memory[idx, i]
                    if np.random.rand() < self.pitch_adjust_rate:
                        self.pitch_adjust_rate = 0.3 + 0.2 * np.random.rand()
                        new_harmony[i] += self.bandwidth * np.random.uniform(-1, 1)
                        new_harmony[i] = np.clip(new_harmony[i], lb[i], ub[i])
                else:
                    new_harmony[i] = lb[i] + (ub[i] - lb[i]) * np.random.rand()

            fitness = func(new_harmony)
            if fitness < best_fitness:
                best_fitness = fitness
                best_solution = new_harmony
                self.best_solutions_history.append(best_fitness)
            
            dynamic_memory_size = max(1, int(self.harmony_memory_size * (1 - np.tanh(eval_count / self.budget))))
            self.harmony_memory_improv_rate = 0.5 + 0.4 * np.cos(np.pi * eval_count / self.budget)
            
            worst_idx = np.argmax([func(h) for h in self.harmony_memory])
            if func(new_harmony) < func(self.harmony_memory[worst_idx]):
                self.harmony_memory[worst_idx] = new_harmony

            if len(self.best_solutions_history) > 10 and np.std(self.best_solutions_history[-10:]) < 0.01:
                if dynamic_memory_size > 1:
                    self.harmony_memory = self.harmony_memory[:dynamic_memory_size]

        return best_solution