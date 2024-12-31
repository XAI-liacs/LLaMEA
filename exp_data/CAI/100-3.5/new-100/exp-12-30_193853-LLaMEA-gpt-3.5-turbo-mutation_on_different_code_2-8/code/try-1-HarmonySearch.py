class HarmonySearch:
    def __init__(self, budget=10000, dim=10, hmcr=0.7, par=0.3, bw=0.01, max_no_improv=100):
        self.budget = budget
        self.dim = dim
        self.hmcr = hmcr
        self.par = par
        self.bw = bw
        self.max_no_improv = max_no_improv
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        harmony_memory = self.initialize_harmony_memory(func)
        no_improv = 0
        
        for i in range(self.budget):
            new_solution = self.generate_new_solution(harmony_memory, func)
            f = func(new_solution)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = new_solution
                no_improv = 0
            else:
                no_improv += 1
            
            if no_improv >= self.max_no_improv:
                break
                
            harmony_memory = self.update_harmony_memory(harmony_memory, new_solution, func)

        return self.f_opt, self.x_opt