class EnhancedAdaptiveHarmonySearch(AdaptiveHarmonySearch):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.bandwidth_factor = 0.1

    def __call__(self, func):
        def generate_new_harmony(harmony_memory):
            new_harmony = np.copy(harmony_memory)
            self.bandwidth *= self.bandwidth_factor
            for i in range(self.dim):
                if np.random.rand() < self.pitch_adjustment_rate:
                    new_harmony[:, i] += np.random.uniform(-self.bandwidth, self.bandwidth, self.harmony_memory_size)
                    new_harmony[:, i] = np.clip(new_harmony[:, i], -5.0, 5.0)
            return new_harmony