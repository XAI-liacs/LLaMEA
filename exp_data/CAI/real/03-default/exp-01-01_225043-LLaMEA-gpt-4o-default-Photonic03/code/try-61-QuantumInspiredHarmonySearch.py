import numpy as np

class QuantumInspiredHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 50
        self.harmonies = np.random.uniform(size=(self.harmony_memory_size, dim))
        self.harmony_memory_fitness = np.full(self.harmony_memory_size, np.inf)
        self.best_harmony = None
        self.best_fitness = np.inf
        self.fitness_evaluations = 0

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        lower_bound, upper_bound = bounds[0], bounds[1]

        def adaptive_pitch_adjustment(evals):
            return 0.1 + 0.9 * np.cos(np.pi * evals / (2 * self.budget))

        def harmony_consideration_rate(evals):
            return 0.5 + 0.4 * np.sin(np.pi * evals / self.budget)

        while self.fitness_evaluations < self.budget:
            new_harmony = np.zeros(self.dim)
            for i in range(self.dim):
                if np.random.rand() < harmony_consideration_rate(self.fitness_evaluations):
                    new_harmony[i] = self.harmonies[np.random.randint(self.harmony_memory_size), i]
                    if np.random.rand() < adaptive_pitch_adjustment(self.fitness_evaluations):
                        new_harmony[i] += np.random.uniform(-1, 1) * (upper_bound[i] - lower_bound[i]) * 0.05
                else:
                    new_harmony[i] = lower_bound[i] + np.random.rand() * (upper_bound[i] - lower_bound[i])

            new_harmony = np.clip(new_harmony, lower_bound, upper_bound)
            new_fitness = func(new_harmony)
            self.fitness_evaluations += 1

            if new_fitness < self.best_fitness:
                self.best_fitness = new_fitness
                self.best_harmony = new_harmony.copy()

            if new_fitness < np.max(self.harmony_memory_fitness):
                worst_idx = np.argmax(self.harmony_memory_fitness)
                self.harmonies[worst_idx] = new_harmony
                self.harmony_memory_fitness[worst_idx] = new_fitness

        return self.best_harmony