import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None

    def mutate(self, population, target_idx):
        candidates = population.copy()
        candidates.remove(target_idx)
        a, b, c = np.random.choice(candidates, 3, replace=False)
        mutant = np.clip(a + self.F * (b - c), -5.0, 5.0)
        return mutant

    def crossover(self, target, trial):
        crossover_points = np.random.rand(self.dim) < self.CR
        offspr = target.copy()
        offspr[crossover_points] = trial[crossover_points]
        return offspr

    def __call__(self, func):
        population = [np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim) for _ in range(10)]
        for _ in range(self.budget):
            for i in range(len(population)):
                target = population[i]
                trial = self.mutate(population, i)
                offspr = self.crossover(target, trial)
                f_target = func(target)
                f_offspr = func(offspr)
                if f_offspr < f_target:
                    population[i] = offspr
                    if f_offspr < self.f_opt:
                        self.f_opt = f_offspr
                        self.x_opt = offspr

        return self.f_opt, self.x_opt