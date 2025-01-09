class EvolutionaryStrategy:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        es = cma.CMAEvolutionStrategy(np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim), 0.5)
        popsize = 10

        for _ in range(self.budget):
            solutions = es.ask(popsize)
            fitness_values = [func(x) for x in solutions]
            es.tell(solutions, fitness_values)
            
            success_rate = es.result.updates / (popsize * self.dim)
            if success_rate < 0.2:
                popsize = max(int(popsize / 2), 2)
            elif success_rate > 0.3:
                popsize = min(popsize * 2, 100)
            
            best_idx = np.argmin(fitness_values)
            if fitness_values[best_idx] < self.f_opt:
                self.f_opt = fitness_values[best_idx]
                self.x_opt = solutions[best_idx]
        
        return self.f_opt, self.x_opt