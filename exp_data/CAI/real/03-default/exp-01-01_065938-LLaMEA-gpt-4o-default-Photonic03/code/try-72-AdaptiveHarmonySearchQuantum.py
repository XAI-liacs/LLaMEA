import numpy as np

class AdaptiveHarmonySearchQuantum:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.hms = 50  # Harmony Memory Size
        self.hmcr = 0.9  # Harmony Memory Consideration Rate
        self.par_min, self.par_max = 0.1, 0.5  # Pitch Adjustment Rate
        self.bandwidth_min, self.bandwidth_max = 0.01, 0.1  # Bandwidth
        self.quantum_factor = 0.05  # Quantum-inspired mutation factor

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        harmony_memory = lb + (ub - lb) * np.random.rand(self.hms, self.dim)
        fitness = np.array([func(harmony) for harmony in harmony_memory])
        best_idx = np.argmin(fitness)
        best_harmony = harmony_memory[best_idx]

        evaluations = self.hms

        while evaluations < self.budget:
            new_harmony = np.zeros(self.dim)

            for i in range(self.dim):
                if np.random.rand() < self.hmcr:
                    idx = np.random.randint(0, self.hms)
                    new_harmony[i] = harmony_memory[idx, i]
                    if np.random.rand() < np.random.uniform(self.par_min, self.par_max):
                        bw = np.random.uniform(self.bandwidth_min, self.bandwidth_max)
                        new_harmony[i] += bw * (np.random.rand() - 0.5)
                else:
                    new_harmony[i] = np.random.uniform(lb[i], ub[i])

            # Quantum-inspired mutation
            if np.random.rand() < self.quantum_factor:
                mutation_scale = np.random.uniform(0.1, 0.3)
                quantum_mutation = mutation_scale * (ub - lb) * np.random.rand(self.dim)
                new_harmony += quantum_mutation

            new_harmony = np.clip(new_harmony, lb, ub)
            new_fitness = func(new_harmony)
            evaluations += 1

            if new_fitness < fitness[best_idx]:
                best_harmony = new_harmony
                best_idx = np.argmax(fitness)
                harmony_memory[best_idx] = new_harmony
                fitness[best_idx] = new_fitness

        return best_harmony