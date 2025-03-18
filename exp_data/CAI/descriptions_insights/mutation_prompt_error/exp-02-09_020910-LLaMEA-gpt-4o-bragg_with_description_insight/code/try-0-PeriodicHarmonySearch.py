import numpy as np

class PeriodicHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.HMS = 10  # Harmony Memory Size
        self.HMCR = 0.9  # Harmony Memory Consideration Rate
        self.PAR = 0.3  # Pitch Adjustment Rate
        self.bandwidth = 0.1  # Bandwidth for pitch adjustment
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub

        # Initialize harmony memory
        harmony_memory = np.random.uniform(lb, ub, (self.HMS, self.dim))
        harmony_fitness = np.array([func(h) for h in harmony_memory])
        self.evaluations += self.HMS

        while self.evaluations < self.budget:
            # Generate new harmony
            new_harmony = np.copy(harmony_memory[np.random.randint(self.HMS)])
            for d in range(self.dim):
                if np.random.rand() < self.HMCR:
                    new_harmony[d] = harmony_memory[np.random.randint(self.HMS)][d]
                    if np.random.rand() < self.PAR:
                        new_harmony[d] += self.bandwidth * (np.random.rand() - 0.5)
                else:
                    new_harmony[d] = np.random.uniform(lb[d], ub[d])

            # Encourage periodicity by rotating the solution
            if np.random.rand() < 0.5:
                period = self.dim // 2
                new_harmony[:period] = new_harmony[period:]

            # Evaluate new harmony
            new_fitness = func(new_harmony)
            self.evaluations += 1

            # Update harmony memory
            if new_fitness > np.min(harmony_fitness):
                worst_index = np.argmin(harmony_fitness)
                harmony_memory[worst_index] = new_harmony
                harmony_fitness[worst_index] = new_fitness

        best_index = np.argmax(harmony_fitness)
        return harmony_memory[best_index]