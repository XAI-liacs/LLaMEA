class HarmonySearch:
    def __init__(self, budget=10000, dim=10, hmcr=0.7, par=0.3):
        self.budget = budget
        self.dim = dim
        self.hmcr = hmcr  # Harmony Memory Consideration Rate
        self.par = par  # Pitch Adjustment Rate
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        hm = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.budget, self.dim))
        
        for i in range(self.budget):
            if np.random.rand() < self.hmcr:
                x_new = hm[np.random.randint(0, self.budget)]
            else:
                x_new = np.mean(hm, axis=0)  # Adaptive HMCR adjustment
            
            for d in range(self.dim):
                if np.random.rand() < self.par:
                    x_new[d] += np.random.uniform(-1, 1) * (x_new[d] - hm[np.random.randint(0, self.budget)][d])
            
            f = func(x_new)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x_new
        
        return self.f_opt, self.x_opt