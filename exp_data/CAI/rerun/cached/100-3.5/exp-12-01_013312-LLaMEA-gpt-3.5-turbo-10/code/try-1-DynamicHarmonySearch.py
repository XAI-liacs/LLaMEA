import numpy as np

class DynamicHarmonySearch(HarmonySearch):
    def generate_new_harmony(self, harmonies, pitch_adjustment_rate=0.3):
        new_harmony = np.copy(harmonies[np.random.randint(0, len(harmonies))])
        for i in range(self.dim):
            if np.random.rand() < pitch_adjustment_rate:
                new_harmony[i] += np.random.normal(scale=0.1) if np.random.rand() < 0.5 else -np.random.normal(scale=0.1)
                new_harmony[i] = np.clip(new_harmony[i], self.lower_bound, self.upper_bound)
        return new_harmony