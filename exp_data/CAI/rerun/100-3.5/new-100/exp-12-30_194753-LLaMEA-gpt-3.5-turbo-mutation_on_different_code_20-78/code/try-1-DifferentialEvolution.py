import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, NP=30, reinit_prob=0.1):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.NP = NP
        self.reinit_prob = reinit_prob
        self.f_opt = np.Inf
        self.x_opt = None
        self.F_lower = 0.4
        self.F_upper = 0.9
        self.CR_lower = 0.6
        self.CR_upper = 0.95

    def __call__(self, func):
        def reinitialize_population():
            return np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.NP, self.dim))

        population = reinitialize_population()
        for i in range(self.budget):
            new_population = []
            self.F = np.clip(np.random.normal(self.F, 0.1), self.F_lower, self.F_upper)
            self.CR = np.clip(np.random.normal(self.CR, 0.1), self.CR_lower, self.CR_upper)
            for j in range(self.NP):
                idxs = np.arange(self.NP)
                np.random.shuffle(idxs)
                r1, r2, r3 = population[np.random.choice(idxs[:3])]
                mutant = population[r1] + self.F * (population[r2] - population[r3])
                cross_points = np.random.rand(self.dim) < self.CR
                trial = np.where(cross_points, mutant, population[j])
                f_trial = func(trial)
                if f_trial < func(population[j]):
                    new_population.append(trial)
                else:
                    new_population.append(population[j])

            population = np.array(new_population)
            best_idx = np.argmin([func(ind) for ind in population])
            if func(population[best_idx]) < self.f_opt:
                self.f_opt = func(population[best_idx])
                self.x_opt = population[best_idx]

            if np.random.rand() < self.reinit_prob:
                population = reinitialize_population()

        return self.f_opt, self.x_opt