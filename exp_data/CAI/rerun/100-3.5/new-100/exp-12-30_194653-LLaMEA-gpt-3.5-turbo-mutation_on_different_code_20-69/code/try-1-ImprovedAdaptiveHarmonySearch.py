class ImprovedAdaptiveHarmonySearch(AdaptiveHarmonySearch):
    def __init__(self, budget=10000, dim=10):
        super().__init__(budget, dim)
        self.bandwidth = 0.1

    def __call__(self, func):
        for i in range(self.budget):
            new_harmony = np.copy(self.x_opt if self.x_opt is not None else np.random.uniform(func.bounds.lb, func.bounds.ub, (self.dim,)))
                    
            for j in range(self.dim):
                if np.random.rand() < self.hmcr:
                    if np.random.rand() < self.par:
                        new_harmony[j] += np.random.uniform(-self.bandwidth, self.bandwidth)
                    else:
                        new_harmony[j] = np.random.uniform(func.bounds.lb, func.bounds.ub)
            
            f = func(new_harmony)
            
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = new_harmony
                self.bandwidth *= 0.9
            else:
                self.bandwidth *= 1.1

        return self.f_opt, self.x_opt