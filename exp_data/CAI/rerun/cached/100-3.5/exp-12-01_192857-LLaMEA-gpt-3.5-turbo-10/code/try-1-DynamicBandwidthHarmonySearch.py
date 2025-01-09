import numpy as np

class DynamicBandwidthHarmonySearch(HarmonySearch):
    def __init__(self, budget, dim, hmcr=0.7, par=0.5, bw=0.01, bw_range=(0.01, 0.1)):
        super().__init__(budget, dim, hmcr, par, bw)
        self.bw_range = bw_range

    def generate_new_harmony(self, harmony_memory):
        new_harmony = np.copy(harmony_memory)
        for i in range(self.dim):
            if np.random.rand() < self.hmcr:
                if np.random.rand() < self.par:
                    rand_index = np.random.randint(self.budget)
                    bw_adjust = np.random.uniform(self.bw_range[0], self.bw_range[1])  # Dynamic bandwidth adjustment
                    new_harmony[rand_index, i] += np.random.uniform(-bw_adjust, bw_adjust)
                    new_harmony[rand_index, i] = np.clip(new_harmony[rand_index, i], -5.0, 5.0)
                else:
                    new_harmony[np.random.randint(self.budget), i] = np.random.uniform(-5.0, 5.0)
        return new_harmony