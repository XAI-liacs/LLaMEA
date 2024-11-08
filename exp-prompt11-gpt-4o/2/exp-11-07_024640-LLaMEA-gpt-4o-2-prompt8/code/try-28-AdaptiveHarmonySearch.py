import numpy as np

class AdaptiveHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.hms = 20  # Harmony Memory Size
        self.hmcr = 0.85  # Harmony Memory Consideration Rate
        self.par = 0.35  # Pitch Adjustment Rate
        self.bandwidth = 0.1  # Bandwidth for pitch adjustment
        self.harmony_memory = None
        self.fitness_memory = None
        self.evaluations = 0

    def initialize_harmony_memory(self):
        self.harmony_memory = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.hms, self.dim))
        self.fitness_memory = np.full(self.hms, float('inf'))

    def evaluate_fitness(self, func, harmony):
        if self.evaluations < self.budget:
            fitness = func(harmony)
            self.evaluations += 1
            return fitness
        return float('inf')

    def improvise_new_harmony(self):
        new_harmony = self.harmony_memory[np.random.randint(self.hms)].copy()
        for i in range(self.dim):
            if np.random.rand() >= self.hmcr:
                new_harmony[i] = np.random.uniform(self.lower_bound, self.upper_bound)
            elif np.random.rand() < self.par:
                adjustment = self.bandwidth * np.random.uniform(-1, 1)
                new_harmony[i] += adjustment
                new_harmony[i] = np.clip(new_harmony[i], self.lower_bound, self.upper_bound)
        return new_harmony

    def update_harmony_memory(self, new_harmony, new_fitness):
        worst_index = np.argmax(self.fitness_memory)
        if new_fitness < self.fitness_memory[worst_index]:
            self.harmony_memory[worst_index] = new_harmony
            self.fitness_memory[worst_index] = new_fitness

    def __call__(self, func):
        self.initialize_harmony_memory()
        for i in range(self.hms):
            self.fitness_memory[i] = self.evaluate_fitness(func, self.harmony_memory[i])

        while self.evaluations < self.budget:
            new_harmony = self.improvise_new_harmony()
            new_fitness = self.evaluate_fitness(func, new_harmony)
            self.update_harmony_memory(new_harmony, new_fitness)

        best_index = np.argmin(self.fitness_memory)
        return self.harmony_memory[best_index], self.fitness_memory[best_index]