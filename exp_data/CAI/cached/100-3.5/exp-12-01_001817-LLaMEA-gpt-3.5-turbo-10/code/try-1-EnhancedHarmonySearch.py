import numpy as np

class EnhancedHarmonySearch(HarmonySearch):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def __call__(self, func):
        def generate_random_solution():
            return np.random.uniform(self.lower_bound, self.upper_bound, self.dim)

        def improvise_harmony(memory, bandwidth):
            new_harmony = []
            for i in range(self.dim):
                if np.random.rand() < bandwidth:
                    new_harmony.append(np.random.uniform(self.lower_bound, self.upper_bound))
                else:
                    idx = np.random.choice(len(memory))
                    new_harmony.append(memory[idx][i])
            return np.array(new_harmony)

        harmony_memory = [generate_random_solution() for _ in range(10)]
        base_bandwidth = 0.5

        for _ in range(self.budget):
            bandwidth = base_bandwidth * (1 - _ / self.budget)  # Adaptive bandwidth control
            new_harmony = improvise_harmony(harmony_memory, bandwidth)
            if func(new_harmony) < min(map(func, harmony_memory)):
                idx = np.argmin(list(map(func, harmony_memory)))
                harmony_memory[idx] = new_harmony

        best_solution = min(harmony_memory, key=func)
        return best_solution