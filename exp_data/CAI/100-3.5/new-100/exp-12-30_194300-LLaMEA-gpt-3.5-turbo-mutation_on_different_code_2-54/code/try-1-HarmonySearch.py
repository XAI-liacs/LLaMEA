class HarmonySearch:
    def __init__(self, budget=10000, dim=10, hmcr=0.7, par=0.4, bw=0.5, adaptive_bw=True, bw_range=(0.1, 1.0)):
        self.budget = budget
        self.dim = dim
        self.hmcr = hmcr  # Harmony Memory Considering Rate
        self.par = par    # Pitch Adjustment Rate
        self.bw = bw      # Bandwidth
        self.adaptive_bw = adaptive_bw  # Adaptive Bandwidth
        self.bw_range = bw_range  # Bandwidth Range
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        # Initialize harmony memory
        harmony_memory = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.dim,))
        
        for _ in range(self.budget):
            # Create a new harmony vector
            new_harmony = np.copy(harmony_memory)
            for i in range(self.dim):
                if np.random.rand() < self.hmcr:
                    new_harmony[i] = harmony_memory[i] if np.random.rand() < self.par else np.random.uniform(func.bounds.lb, func.bounds.ub)
                    new_harmony[i] += np.random.uniform(-self.bw, self.bw)
                else:
                    new_harmony[i] = np.random.uniform(func.bounds.lb, func.bounds.ub)
            
            if self.adaptive_bw:  # Adaptive Bandwidth logic
                current_bw = self.bw_range[0] + ((_ + 1) / self.budget) * (self.bw_range[1] - self.bw_range[0])
                self.bw = current_bw
            
            # Evaluate the new harmony
            f = func(new_harmony)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = new_harmony
                harmony_memory = np.copy(new_harmony)
                
        return self.f_opt, self.x_opt