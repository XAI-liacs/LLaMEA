class FireworkAlgorithm:
    def __init__(self, budget=10000, dim=10, n_fireworks=10, n_sparks=5, max_amp=0.5, amp_decay=0.9):
        self.budget = budget
        self.dim = dim
        self.n_fireworks = n_fireworks
        self.n_sparks = n_sparks
        self.max_amp = max_amp
        self.amp_decay = amp_decay
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        fireworks = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.n_fireworks, self.dim))
        
        for _ in range(self.budget):
            for fw in fireworks:
                sparks = fw + np.random.uniform(-self.max_amp, self.max_amp, size=(self.n_sparks, self.dim))
                f_sparks = [func(sp) for sp in sparks]
                min_idx = np.argmin(f_sparks)
                if f_sparks[min_idx] < self.f_opt:
                    self.f_opt = f_sparks[min_idx]
                    self.x_opt = sparks[min_idx]
            self.max_amp *= self.amp_decay

        return self.f_opt, self.x_opt