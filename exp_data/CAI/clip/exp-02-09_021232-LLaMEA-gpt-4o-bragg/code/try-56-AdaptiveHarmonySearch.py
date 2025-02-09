import numpy as np

class AdaptiveHarmonySearch:
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

    def initialize_harmony_memory(self, bounds):
        self.harmony_memory = np.random.uniform(bounds.lb, bounds.ub, (self.num_harmonies, self.dim))

    def evaluate(self, func, harmony):
        fitness = func(harmony)
        if fitness < self.best_fitness:
            self.best_fitness = fitness
            self.best_solution = harmony.copy()  # Ensure copy of the array is stored
        return fitness

    def update_harmony_memory(self, new_harmony, new_fitness):
        if new_fitness < self.best_fitness or np.random.rand() < 0.25:  # Adjusted acceptance probability
            worst_idx = np.argmax([self.evaluate(lambda x: np.sum(x), h) for h in self.harmony_memory])
            self.harmony_memory[worst_idx] = new_harmony  # Direct replacement without extra check

    def generate_new_harmony(self, bounds):
        new_harmony = np.zeros(self.dim)
        for i in range(self.dim):
            if np.random.rand() < self.HMCR:
                new_harmony[i] = np.mean(self.harmony_memory[:, i])  # Mean diversity introduction
                if np.random.rand() < self.PAR:
                    new_harmony[i] += np.random.normal(0, self.bandwidth)  # Adjusted from uniform to normal distribution
            else:
                new_harmony[i] = np.random.uniform(bounds.lb[i], bounds.ub[i])
        return np.clip(new_harmony, bounds.lb, bounds.ub)

    def adjust_parameters(self):
        improvement_rate = min(1.0, abs(self.best_fitness / 100))  # Absolute dynamic adjustment
        self.bandwidth *= np.random.uniform(0.8, 1.2 * improvement_rate)  # Stochastic bandwidth adjustment
        self.PAR = min(1.0, self.PAR + np.random.uniform(0.03, 0.05))  # Adjust pitch rate slightly
        self.num_harmonies = max(5, min(15, int(self.num_harmonies * np.random.uniform(0.95, 1.05))))  # Tighter adjustment

    def __call__(self, func):
        bounds = func.bounds
        self.initialize_harmony_memory(bounds)

        evaluations = 0
        while evaluations < self.budget:
            new_harmony = self.generate_new_harmony(bounds)
            new_fitness = self.evaluate(func, new_harmony)
            self.update_harmony_memory(new_harmony, new_fitness)
            self.adjust_parameters()
            evaluations += 1

        return self.best_solution