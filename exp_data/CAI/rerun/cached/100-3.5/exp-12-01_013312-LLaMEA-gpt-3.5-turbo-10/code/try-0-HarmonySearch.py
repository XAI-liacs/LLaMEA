import numpy as np

class HarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def generate_initial_harmony(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, self.dim)

    def generate_new_harmony(self, harmonies, pitch_adjustment_rate=0.3):
        new_harmony = np.copy(harmonies[np.random.randint(0, len(harmonies))])
        for i in range(self.dim):
            if np.random.rand() < pitch_adjustment_rate:
                new_harmony[i] = np.random.uniform(self.lower_bound, self.upper_bound)
        return new_harmony

    def __call__(self, func):
        harmonies = [self.generate_initial_harmony() for _ in range(10)]
        for _ in range(self.budget):
            new_harmony = self.generate_new_harmony(harmonies)
            if func(new_harmony) < func(harmonies[np.argmax([func(h) for h in harmonies])]):
                harmonies[np.argmax([func(h) for h in harmonies])] = new_harmony
        return harmonies[np.argmin([func(h) for h in harmonies])]