class HarmonySearch:
    def __init__(self, budget=10000, dim=10, hmcr=0.7, par=0.3, bw=0.01, local_search_prob=0.1, local_search_range=0.1):
        self.budget = budget
        self.dim = dim
        self.hmcr = hmcr  # Harmony Memory Considering Rate
        self.par = par    # Pitch Adjustment Rate
        self.bw = bw      # Bandwidth
        self.local_search_prob = local_search_prob
        self.local_search_range = local_search_range
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        harmony_memory = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.dim,))
        for i in range(self.budget):
            new_solution = np.zeros((self.dim,))
            for j in range(self.dim):
                if np.random.rand() < self.hmcr:
                    new_solution[j] = harmony_memory[j]
                else:
                    rand_index = np.random.randint(self.dim)
                    new_solution[j] = harmony_memory[rand_index]
                    if np.random.rand() < self.par:
                        new_solution[j] = new_solution[j] + self.bw * np.random.uniform(func.bounds.lb, func.bounds.ub)
                
                if np.random.rand() < self.local_search_prob and i > 0:
                    local_search_solution = self.x_opt + self.local_search_range * np.random.randn(self.dim)
                    local_search_solution = np.clip(local_search_solution, func.bounds.lb, func.bounds.ub)
                    local_search_fitness = func(local_search_solution)
                    if local_search_fitness < self.f_opt:
                        self.f_opt = local_search_fitness
                        self.x_opt = local_search_solution
                        harmony_memory = np.copy(local_search_solution)
            
            f = func(new_solution)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = new_solution
                harmony_memory = np.copy(new_solution)
            
        return self.f_opt, self.x_opt