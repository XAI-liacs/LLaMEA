import numpy as np

class EnhancedDynamicPitchHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pitch_range = 0.1

    def __call__(self, func):
        def initialize_harmony():
            return np.random.uniform(self.lower_bound, self.upper_bound, self.dim)

        def adjust_value(value):
            return np.clip(value, self.lower_bound, self.upper_bound)

        def adjust_pitch(pitch):
            return max(0.001, pitch * np.exp(np.random.uniform(-1, 1)))

        def opposition_based_learning(value):
            return self.lower_bound + self.upper_bound - value

        def mutation(harmony):
            mutated_index = np.random.randint(self.dim)
            mutation_amount = np.random.uniform(-self.pitch_range, self.pitch_range)
            harmony[mutated_index] += mutation_amount
            harmony[mutated_index] = adjust_value(harmony[mutated_index])
            return harmony

        def harmony_search():
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

                new_harmony = mutation(new_harmony)

                new_harmony_opposite = opposition_based_learning(new_harmony)
                new_fitness_opposite = func(new_harmony_opposite)

                if new_fitness_opposite < best_fitness:
                    best_solution = np.copy(new_harmony_opposite)
                    best_fitness = new_fitness_opposite

                harmony_memory[np.argmax([func(h) for h in harmony_memory])] = new_harmony
                pitch = adjust_pitch(pitch)

            return best_solution

        return harmony_search()