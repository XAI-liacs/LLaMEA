class HarmonySearch:
    def __init__(self, budget=10000, dim=10, harmony_memory_size=10, bandwidth=0.01, pitch_adjust_rate=0.5, memory_consideration=0.9):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = harmony_memory_size
        self.bandwidth = bandwidth
        self.pitch_adjust_rate = pitch_adjust_rate
        self.memory_consideration = memory_consideration
        self.f_opt = np.Inf
        self.x_opt = None
    
    def generate_new_harmony(self, func, harmony_memory):
        new_harmony = np.zeros(self.dim)
        for i in range(self.dim):
            if np.random.rand() < self.pitch_adjust_rate:
                random_index = np.random.randint(0, len(harmony_memory))
                new_harmony[i] = harmony_memory[random_index][i] + np.random.uniform(-self.bandwidth, self.bandwidth)
            else:
                new_harmony[i] = np.random.uniform(func.bounds.lb, func.bounds.ub)
            if np.random.rand() < self.memory_consideration:
                random_index = np.random.randint(0, len(harmony_memory))
                new_harmony[i] = harmony_memory[random_index][i]
        return new_harmony