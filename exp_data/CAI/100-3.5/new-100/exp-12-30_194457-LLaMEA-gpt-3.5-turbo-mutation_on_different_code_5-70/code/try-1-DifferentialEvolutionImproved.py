class DifferentialEvolutionImproved(DifferentialEvolution):
    def __init__(self, budget=10000, dim=10, F=None, CR=None, pop_size=30):
        super().__init__(budget, dim, F, CR, pop_size)
        
    def __call__(self, func):
        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        F = np.random.uniform(0, 2, size=self.pop_size) if self.F is None else np.full(self.pop_size, self.F)
        CR = np.random.uniform(0, 1, size=self.pop_size) if self.CR is None else np.full(self.pop_size, self.CR)
        
        for _ in range(self.budget):
            for i in range(self.pop_size):
                idxs = list(range(self.pop_size))
                idxs.remove(i)
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant = pop[a] + F[i] * (pop[b] - pop[c])
                j_rand = np.random.randint(self.dim)
                trial = np.copy(pop[i])
                for j in range(self.dim):
                    if np.random.rand() < CR[i] or j == j_rand:
                        trial[j] = mutant[j]
                f_trial = func(trial)
                if f_trial < func(pop[i]):
                    pop[i] = trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = np.copy(trial)
                        
            F = F + 0.01 * (0.5 - np.random.rand())  # Adaptive control for F
            F = np.clip(F, 0, 2)
            CR = CR + 0.01 * (0.9 - np.random.rand())  # Adaptive control for CR
            CR = np.clip(CR, 0, 1)
        
        return self.f_opt, self.x_opt