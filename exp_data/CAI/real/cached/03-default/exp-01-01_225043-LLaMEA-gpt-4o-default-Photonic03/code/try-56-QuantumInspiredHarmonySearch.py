import numpy as np

class QuantumInspiredHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 50
        self.harmony_memory = np.random.uniform(size=(self.harmony_memory_size, dim))
        self.harmony_fitness = np.full(self.harmony_memory_size, np.inf)
        self.best_harmony = None
        self.best_fitness = np.inf
        self.fitness_evaluations = 0

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        lower_bound, upper_bound = bounds[0], bounds[1]

        memory_consideration_rate = 0.95
        pitch_adjustment_rate = lambda evals: 0.4 + 0.5 * (1 - evals / self.budget)
        bandwidth = lambda evals: (upper_bound - lower_bound) * (1 - evals / self.budget)

        for i in range(self.harmony_memory_size):
            self.harmony_fitness[i] = func(self.harmony_memory[i])
            self.fitness_evaluations += 1
            if self.harmony_fitness[i] < self.best_fitness:
                self.best_fitness = self.harmony_fitness[i]
                self.best_harmony = self.harmony_memory[i]

        while self.fitness_evaluations < self.budget:
            new_harmony = np.empty(self.dim)
            for j in range(self.dim):
                if np.random.rand() < memory_consideration_rate:
                    new_harmony[j] = self.harmony_memory[np.random.randint(self.harmony_memory_size)][j]
                    if np.random.rand() < pitch_adjustment_rate(self.fitness_evaluations):
                        new_harmony[j] += np.random.uniform(-1, 1) * bandwidth(self.fitness_evaluations)[j]
                else:
                    new_harmony[j] = lower_bound[j] + np.random.rand() * (upper_bound[j] - lower_bound[j])

            new_harmony = np.clip(new_harmony, lower_bound, upper_bound)
            new_fitness = func(new_harmony)
            self.fitness_evaluations += 1

            if new_fitness < self.best_fitness:
                self.best_fitness = new_fitness
                self.best_harmony = new_harmony.copy()

            worst_idx = np.argmax(self.harmony_fitness)
            if new_fitness < self.harmony_fitness[worst_idx]:
                self.harmony_memory[worst_idx] = new_harmony
                self.harmony_fitness[worst_idx] = new_fitness

        return self.best_harmony