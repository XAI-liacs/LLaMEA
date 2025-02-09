import numpy as np

class EnhancedAdaptiveHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.HMCR = 0.9  # Harmony Memory Consideration Rate
        self.PAR = 0.3   # Pitch Adjustment Rate
        self.bandwidth = 0.05  # Initial Bandwidth
        self.num_harmonies = 10  # Number of harmonies in the memory
        self.harmony_memory = None
        self.best_solution = None
        self.best_fitness = float('inf')
        self.elite_solution = None
        self.elite_fitness = float('inf')

    def initialize_harmony_memory(self, bounds):
        self.harmony_memory = np.random.uniform(bounds.lb, bounds.ub, (self.num_harmonies, self.dim))

    def evaluate(self, func, harmony):
        fitness = func(harmony)
        if fitness < self.best_fitness:
            self.best_fitness = fitness
            self.best_solution = harmony.copy()
        if fitness < self.elite_fitness and np.random.rand() < 0.1:  # Ensure occasional update of elite
            self.elite_fitness = fitness
            self.elite_solution = harmony.copy()
        return fitness

    def update_harmony_memory(self, new_harmony, new_fitness):
        if new_fitness < self.best_fitness or np.random.rand() < 0.3:
            worst_idx = np.argmax([self.evaluate(lambda x: np.sum(x), h) for h in self.harmony_memory])
            if new_fitness < self.evaluate(lambda x: np.sum(x), self.harmony_memory[worst_idx]):
                self.harmony_memory[worst_idx] = new_harmony

    def generate_new_harmony(self, bounds):
        new_harmony = np.zeros(self.dim)
        for i in range(self.dim):
            if np.random.rand() < self.HMCR:
                new_harmony[i] = self.harmony_memory[np.random.randint(self.num_harmonies), i]
                if np.random.rand() < self.PAR:
                    new_harmony[i] += np.random.normal(0, self.bandwidth)
            else:
                new_harmony[i] = np.random.uniform(bounds.lb[i], bounds.ub[i])
        return np.clip(new_harmony, bounds.lb, bounds.ub)

    def adjust_parameters(self, evaluations):
        self.bandwidth *= np.exp(-evaluations / (0.1 * self.budget))
        self.PAR = min(1.0, self.PAR + np.exp(-evaluations / self.budget))
        if evaluations % (self.budget // 10) == 0:
            self.harmony_memory[np.random.randint(self.num_harmonies)] = self.elite_solution.copy()

    def __call__(self, func):
        bounds = func.bounds
        self.initialize_harmony_memory(bounds)

        evaluations = 0
        while evaluations < self.budget:
            new_harmony = self.generate_new_harmony(bounds)
            new_fitness = self.evaluate(func, new_harmony)
            self.update_harmony_memory(new_harmony, new_fitness)
            self.adjust_parameters(evaluations)
            evaluations += 1

        return self.best_solution