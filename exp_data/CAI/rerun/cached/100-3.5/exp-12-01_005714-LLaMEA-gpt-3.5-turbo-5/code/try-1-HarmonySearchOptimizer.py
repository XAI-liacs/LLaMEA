import numpy as np

class HarmonySearchOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.bandwidth = 0.01

    def __call__(self, func):
        def generate_harmony():
            return np.random.uniform(self.lower_bound, self.upper_bound, self.dim)

        def improvise_harmony(memory, pitch_adjustment_rate):
            new_harmony = np.array([max(min(note + np.random.uniform(-pitch_adjustment_rate, pitch_adjustment_rate), self.upper_bound), self.lower_bound) for note in memory])
            return new_harmony

        harmonies = [generate_harmony() for _ in range(5)]
        fitness_values = [func(harmony) for harmony in harmonies]

        for _ in range(self.budget - 5):
            pitch_adjustment_rate = (1.0 - ((_) / (self.budget - 5))) * self.bandwidth
            new_harmony = improvise_harmony(harmonies[np.argmax(fitness_values)], pitch_adjustment_rate)
            new_fitness = func(new_harmony)

            if new_fitness < max(fitness_values):
                index_to_replace = np.argmax(fitness_values)
                harmonies[index_to_replace] = new_harmony
                fitness_values[index_to_replace] = new_fitness

        return harmonies[np.argmin(fitness_values)]