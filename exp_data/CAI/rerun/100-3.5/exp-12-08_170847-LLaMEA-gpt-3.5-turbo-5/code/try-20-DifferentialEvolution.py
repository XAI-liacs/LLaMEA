class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, pop_size=50):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        conv_rate = 0.0

        for _ in range(self.budget):
            for i in range(self.pop_size):
                idxs = np.random.choice(self.pop_size, size=3, replace=False)
                a, b, c = pop[idxs]

                R = np.random.randint(self.dim)
                trial = np.copy(pop[i])
                for j in range(self.dim):
                    if j == R or np.random.rand() < self.CR:
                        trial[j] = a[j] + self.F * (b[j] - c[j])

                trial_fit = func(trial)
                if trial_fit < func(pop[i]):
                    pop[i] = trial

                    if trial_fit < self.f_opt:
                        self.f_opt = trial_fit
                        self.x_opt = trial

            current_best = np.min([func(ind) for ind in pop])
            if current_best < self.f_opt:
                self.f_opt = current_best
                self.x_opt = pop[np.argmin([func(ind) for ind in pop])]

            conv_rate = 0.9 * conv_rate + 0.1 * (current_best - self.f_opt)
            if conv_rate < 1e-6:  # Dynamic population size adaptation
                self.pop_size = int(1.1 * self.pop_size)
                pop = np.vstack((pop, np.random.uniform(func.bounds.lb, func.bounds.ub, size=(int(0.1*self.pop_size), self.dim)))

        return self.f_opt, self.x_opt