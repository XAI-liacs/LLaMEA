import numpy as np

class QuantumHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 50
        self.hmcr = 0.9
        self.par = 0.3
        self.bw = 0.05
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
                if np.random.rand() < self.hmcr:
                    idx = np.random.randint(self.harmony_memory_size)
                    new_harmony[i] = harmony_memory[idx, i]
                    if np.random.rand() < self.par:
                        delta = np.random.uniform(-self.bw, self.bw)
                        new_harmony[i] += delta * (ub[i] - lb[i])
                else:
                    new_harmony[i] = np.random.uniform(lb[i], ub[i])

            new_harmony = np.clip(new_harmony, lb, ub)
            new_fitness = func(new_harmony)
            evaluations += 1

            if new_fitness < fitness[best_idx]:
                worst_idx = np.argmax(fitness)
                harmony_memory[worst_idx] = new_harmony
                fitness[worst_idx] = new_fitness
                best_idx = np.argmin(fitness)
                best_global = harmony_memory[best_idx]

            self.history.append(best_global)

            # Adaptively adjusting parameters
            self.hmcr = np.clip(self.hmcr + 0.05 * (np.random.rand() - 0.5), 0.7, 0.95)
            self.par = np.clip(self.par + 0.05 * (np.random.rand() - 0.5), 0.1, 0.5)
            self.bw = np.clip(self.bw + 0.01 * (np.random.rand() - 0.5), 0.01, 0.1)

        return best_global