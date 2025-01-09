import numpy as np

class QuantumHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 10
        self.HMCR = 0.7
        self.PAR = 0.3
        self.bandwidth = 0.01
        self.pitch_adjust_rate = 0.5

    def __call__(self, func):
        def initialize_harmony_memory():
            return np.random.uniform(-5.0, 5.0, (self.harmony_memory_size, self.dim))

        def update_pitch_adjust_rate(iter_count):
            return self.pitch_adjust_rate * np.exp(-0.1 * iter_count)

        def harmony_search():
            harmony_memory = initialize_harmony_memory()
            best_solution = None
            best_fitness = np.inf

            for _ in range(self.budget):
                new_harmony = np.zeros((1, self.dim))
                for i in range(self.dim):
                    if np.random.rand() < self.HMCR:
                        idx = np.random.randint(self.harmony_memory_size)
                        new_harmony[0, i] = harmony_memory[idx, i]
                    else:
                        new_harmony[0, i] = np.random.uniform(-5.0, 5.0)

                    if np.random.rand() < self.PAR:
                        new_harmony[0, i] += np.random.normal(0, self.bandwidth)

                new_fitness = func(new_harmony.reshape(self.dim))
                if new_fitness < best_fitness:
                    best_solution = new_harmony
                    best_fitness = new_fitness

                if np.random.rand() < update_pitch_adjust_rate(_):
                    self.bandwidth *= np.exp(-0.1 * _)

                harmony_memory[np.argmax(func(harmony_memory.reshape(self.harmony_memory_size, self.dim)))] = new_harmony

            return best_solution, best_fitness

        return harmony_search()