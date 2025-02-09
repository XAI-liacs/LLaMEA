import numpy as np

class QuantumHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 10
        self.harmony_memory = np.random.rand(self.harmony_memory_size, self.dim)
        self.harmony_memory_improv_rate = 0.9
        self.pitch_adjust_rate = 0.3
        self.bandwidth = 0.05

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.harmony_memory = lb + (ub - lb) * self.harmony_memory

        best_solution = None
        best_fitness = float('inf')
        fitness_memory = np.array([func(h) for h in self.harmony_memory])

        for _ in range(self.budget):
            sorted_indices = np.argsort(fitness_memory)
            improv_rate_adaptive = self.harmony_memory_improv_rate * (1 - sorted_indices / self.harmony_memory_size)
            improv_rate_adaptive = 1 - improv_rate_adaptive
            pitch_adjust_adaptive = self.pitch_adjust_rate * (sorted_indices / self.harmony_memory_size)

            memory_weights = (1 / (1 + fitness_memory)) / np.sum(1 / (1 + fitness_memory))
            new_harmony = np.zeros(self.dim)
            for i in range(self.dim):
                if np.random.rand() < improv_rate_adaptive[i]:
                    idx = np.random.choice(np.arange(self.harmony_memory_size), p=memory_weights)
                    new_harmony[i] = self.harmony_memory[idx, i]
                    
                    if np.random.rand() < pitch_adjust_adaptive[i]:
                        self.bandwidth = 0.01 + 0.04 * np.random.rand()
                        new_harmony[i] += self.bandwidth * np.random.uniform(-1, 1)
                        new_harmony[i] = np.clip(new_harmony[i], lb[i], ub[i])
                else:
                    new_harmony[i] = lb[i] + (ub[i] - lb[i]) * np.random.rand()

            fitness = func(new_harmony)
            if fitness < best_fitness:
                best_fitness = fitness
                best_solution = new_harmony

            worst_idx = np.argmax(fitness_memory)
            if fitness < fitness_memory[worst_idx]:
                self.harmony_memory[worst_idx] = new_harmony
                fitness_memory[worst_idx] = fitness

        return best_solution