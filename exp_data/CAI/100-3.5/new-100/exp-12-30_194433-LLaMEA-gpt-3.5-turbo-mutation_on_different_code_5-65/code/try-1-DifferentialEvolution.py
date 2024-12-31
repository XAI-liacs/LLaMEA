class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, pop_size=20, adapt_scale=True):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None
        self.adapt_scale = adapt_scale

    def mutate(self, pop, target_idx):
        candidates = [idx for idx in range(len(pop)) if idx != target_idx]
        a, b, c = np.random.choice(candidates, 3, replace=False)
        scale_factor = np.random.normal(1.0, 0.1) if self.adapt_scale else self.F
        mutant = pop[a] + scale_factor * (pop[b] - pop[c])
        return np.clip(mutant, -5.0, 5.0)