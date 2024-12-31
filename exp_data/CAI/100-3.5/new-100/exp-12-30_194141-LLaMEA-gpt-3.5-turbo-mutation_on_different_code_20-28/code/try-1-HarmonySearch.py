class HarmonySearch:
    def __init__(self, budget=10000, dim=10, hmcr=0.7, par=0.3, bw=0.01, bw_range=(0.01, 0.2), bw_decay=0.95):
        self.budget = budget
        self.dim = dim
        self.hmcr = hmcr  # Harmony Memory Considering Rate
        self.par = par    # Pitch Adjustment Rate
        self.bw = bw      # Bandwidth
        self.bw_range = bw_range  # Range for bandwidth adjustment
        self.bw_decay = bw_decay  # Bandwidth decay factor
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        harmony_memory = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        for i in range(self.budget):
            new_harmony = np.zeros(self.dim)
            for j in range(self.dim):
                if np.random.rand() < self.hmcr:
                    new_harmony[j] = harmony_memory[np.random.randint(self.budget)][j]
                else:
                    new_harmony[j] = np.random.uniform(func.bounds.lb, func.bounds.ub)
                    if np.random.rand() < self.par:
                        new_harmony[j] += np.random.uniform(-self.bw, self.bw)

            f = func(new_harmony)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = new_harmony

            # Adaptive adjustment of bandwidth
            self.bw = max(self.bw_range[0], self.bw * self.bw_decay) if f < self.f_opt else min(self.bw_range[1], self.bw * (1/self.bw_decay))

        return self.f_opt, self.x_opt