import numpy as np

class AQHS:
    def __init__(self, budget, dim, harmony_size=20, hmcr=0.9, par=0.3, bw=0.01, quantum_prob=0.1):
        self.budget = budget
        self.dim = dim
        self.harmony_size = harmony_size
        self.hmcr = hmcr
        self.par = par
        self.bw = bw
        self.quantum_prob = quantum_prob
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        harmony_memory = self.initialize_harmony_memory(lb, ub)
        best_harmony = min(harmony_memory, key=func)
        best_value = func(best_harmony)
        self.evaluations += len(harmony_memory)

        while self.evaluations < self.budget:
            new_harmony = np.zeros(self.dim)
            for i in range(self.dim):
                if np.random.rand() < self.hmcr:
                    new_harmony[i] = harmony_memory[np.random.randint(self.harmony_size)][i]
                    if np.random.rand() < self.par:
                        new_harmony[i] += self.bw * (np.random.rand() - 0.5)
                else:
                    new_harmony[i] = np.random.uniform(lb[i], ub[i])

                if np.random.rand() < self.quantum_prob:
                    new_harmony[i] = self.quantum_perturbation(new_harmony[i], lb[i], ub[i])

            new_harmony = np.clip(new_harmony, lb, ub)
            new_value = func(new_harmony)
            self.evaluations += 1

            if new_value < best_value:
                best_value = new_value
                best_harmony = new_harmony

            worst_index = max(range(self.harmony_size), key=lambda idx: func(harmony_memory[idx]))
            if new_value < func(harmony_memory[worst_index]):
                harmony_memory[worst_index] = new_harmony

            if self.evaluations >= self.budget:
                break

        return best_harmony

    def initialize_harmony_memory(self, lb, ub):
        return [np.random.uniform(lb, ub, self.dim) for _ in range(self.harmony_size)]

    def quantum_perturbation(self, position, lb, ub):
        q_position = position + (np.random.rand() - 0.5) * (ub - lb) * 0.1
        return np.clip(q_position, lb, ub)