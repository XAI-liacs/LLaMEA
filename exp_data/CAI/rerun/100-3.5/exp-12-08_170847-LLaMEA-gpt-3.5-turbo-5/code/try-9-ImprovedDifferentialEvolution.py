import numpy as np

class ImprovedDifferentialEvolution(DifferentialEvolution):
    def __call__(self, func):
        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))

        for _ in range(self.budget):
            new_pop = np.copy(pop)

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
                    new_pop[i] = trial

                    if trial_fit < self.f_opt:
                        self.f_opt = trial_fit
                        self.x_opt = trial

            fitness = np.array([func(ind) for ind in new_pop])
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < func(pop[i]):
                pop[i] = new_pop[best_idx]

        return self.f_opt, self.x_opt