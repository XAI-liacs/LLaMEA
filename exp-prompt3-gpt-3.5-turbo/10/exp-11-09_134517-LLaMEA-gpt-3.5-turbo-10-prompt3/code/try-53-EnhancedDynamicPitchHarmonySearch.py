import numpy as np

class EnhancedDynamicPitchHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pitch_range = 0.1
        self.pitch_adapt_rate = 0.1
        self.bandwidth = 0.6

    def __call__(self, func):
        def initialize_harmony():
            return np.random.uniform(self.lower_bound, self.upper_bound, self.dim)

        def adjust_value(value):
            return np.clip(value, self.lower_bound, self.upper_bound)

        def adjust_pitch(pitch, improvement):
            return max(0.001, pitch * np.exp(self.pitch_adapt_rate * improvement))

        def adjust_bandwidth(bandwidth, improvement):
            return min(0.9, bandwidth * np.exp(self.pitch_adapt_rate * improvement))

        def opposition_based_learning(value):
            return self.lower_bound + self.upper_bound - value

        def enhanced_harmony_search():
            harmony_memory = [initialize_harmony() for _ in range(self.budget)]
            best_solution = np.copy(harmony_memory[0])
            best_fitness = func(best_solution)
            pitch = self.pitch_range

            for _ in range(self.budget):
                new_harmony = np.mean(harmony_memory, axis=0)
                new_harmony = adjust_value(new_harmony)
                new_fitness = func(new_harmony)

                if new_fitness < best_fitness:
                    best_solution = np.copy(new_harmony)
                    best_fitness = new_fitness
                    pitch = adjust_pitch(pitch, 1)

                for i in range(self.budget):
                    harmony_memory[i] = opposition_based_learning(harmony_memory[i])

                index = np.random.randint(self.dim)
                new_harmony[index] = np.random.uniform(max(self.lower_bound, new_harmony[index] - pitch),
                                                       min(self.upper_bound, new_harmony[index] + pitch))

                new_harmony_opposite = opposition_based_learning(new_harmony)
                new_fitness_opposite = func(new_harmony_opposite)

                if new_fitness_opposite < best_fitness:
                    best_solution = np.copy(new_harmony_opposite)
                    best_fitness = new_fitness_opposite
                    pitch = adjust_pitch(pitch, 1)

                harmony_memory[np.argmax([func(h) for h in harmony_memory])] = new_harmony

            return best_solution

        return enhanced_harmony_search()