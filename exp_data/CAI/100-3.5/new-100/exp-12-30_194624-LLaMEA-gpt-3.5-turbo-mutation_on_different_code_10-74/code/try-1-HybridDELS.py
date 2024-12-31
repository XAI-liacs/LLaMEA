class HybridDELS:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.mutation_factor = 0.5  # Initialize mutation factor

    def __call__(self, func):
        def local_search(x):
            res = minimize(func, x, bounds=[(-5.0, 5.0)]*self.dim, method='L-BFGS-B')
            return res.fun, res.x
        
        pop_size = 10
        crossover_prob = 0.7
        max_iter = self.budget // pop_size
        
        pop = np.random.uniform(-5.0, 5.0, size=(pop_size, self.dim))
        
        for _ in range(max_iter):
            for i in range(pop_size):
                target = pop[i]
                
                r1, r2, r3 = np.random.choice(pop, 3, replace=False)
                mutant = np.clip(r1 + self.mutation_factor * (r2 - r3), -5.0, 5.0)
                
                crossover_mask = np.random.rand(self.dim) < crossover_prob
                trial = np.where(crossover_mask, mutant, target)
                
                trial_f = func(trial)
                if trial_f < self.f_opt:
                    self.f_opt = trial_f
                    self.x_opt = trial
                
                if func(target) > trial_f:
                    pop[i] = trial
                
                if func(target) > self.f_opt:
                    pop[i] = self.x_opt
                
                local_f, local_x = local_search(target)
                if local_f < self.f_opt:
                    self.f_opt = local_f
                    self.x_opt = local_x
            
            # Adaptive control of mutation factor based on function landscape
            best_val = np.min([func(ind) for ind in pop])
            if best_val < self.f_opt:
                self.mutation_factor /= 2
            else:
                self.mutation_factor *= 2
                self.mutation_factor = min(self.mutation_factor, 2.0)  # Cap at 2.0
        
        return self.f_opt, self.x_opt