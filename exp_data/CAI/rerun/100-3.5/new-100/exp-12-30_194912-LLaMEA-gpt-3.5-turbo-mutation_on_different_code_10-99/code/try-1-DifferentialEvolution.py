class DifferentialEvolution:
    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        for _ in range(self.budget):
            for i in range(self.budget):
                idxs = [idx for idx in range(self.budget) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                F_i = np.random.normal(self.F, 0.1)  # Dynamic mutation scaling
                mutant = population[i] + F_i * (a - b)
                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, mutant, population[i])
                
                f_trial = func(trial)
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial
                    population[i] = trial
        return self.f_opt, self.x_opt