import numpy as np

class HarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, size=(self.budget, self.dim))
        harmony_memory_fit = np.array([func(x) for x in harmony_memory])

        for _ in range(self.budget - len(harmony_memory)):
            new_harmony = np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim)
            new_fit = func(new_harmony)

            if new_fit < np.max(harmony_memory_fit):
                index = np.argmax(harmony_memory_fit)
                harmony_memory[index] = new_harmony
                harmony_memory_fit[index] = new_fit

        best_index = np.argmin(harmony_memory_fit)
        return harmony_memory[best_index]