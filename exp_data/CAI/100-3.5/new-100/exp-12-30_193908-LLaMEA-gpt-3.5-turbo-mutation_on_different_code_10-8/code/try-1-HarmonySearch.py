class HarmonySearch:
    def __init__(self, budget=10000, dim=10, hmcr=0.7, par=0.5, bw=0.01, bw_decay=0.99):
        self.budget = budget
        self.dim = dim
        self.hmcr = hmcr  # Harmony Memory Considering Rate
        self.par = par    # Pitch Adjustment Rate
        self.bw = bw      # Bandwidth
        self.bw_decay = bw_decay  # Bandwidth Decay Rate
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        # Initialize harmony memory
        harmony_memory = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.dim,))
        
        for i in range(self.budget):
            # Create new harmony
            new_harmony = np.zeros(self.dim)
            for j in range(self.dim):
                if np.random.rand() < self.hmcr:
                    new_harmony[j] = harmony_memory[j]
                else:
                    new_harmony[j] = np.random.uniform(func.bounds.lb, func.bounds.ub)
                
                if np.random.rand() < self.par:
                    new_harmony[j] += np.random.uniform(-self.bw, self.bw)
                    
                new_harmony[j] = np.clip(new_harmony[j], func.bounds.lb, func.bounds.ub)
                
            # Evaluate new harmony
            f = func(new_harmony)
            
            # Update harmony memory
            if f < func(harmony_memory):
                harmony_memory = new_harmony.copy()
            
            # Update global optimum
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = new_harmony.copy()
                
            # Update bandwidth
            self.bw *= self.bw_decay
            
        return self.f_opt, self.x_opt