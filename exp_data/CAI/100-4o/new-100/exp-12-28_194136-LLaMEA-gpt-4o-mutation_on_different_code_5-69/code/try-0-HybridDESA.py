import numpy as np

class HybridDESA:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.pop_size = 10 * dim
        self.F = 0.5  # Differential weight
        self.CR = 0.7  # Crossover probability
        self.temperature = 1.0
        self.cooling_rate = 0.99

    def de_mutation(self, pop, idx):
        indices = [i for i in range(self.pop_size) if i != idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = pop[a] + self.F * (pop[b] - pop[c])
        return np.clip(mutant, -5.0, 5.0)

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def __call__(self, func):
        bounds = func.bounds
        pop = np.random.uniform(bounds.lb, bounds.ub, (self.pop_size, self.dim))
        fit = np.array([func(ind) for ind in pop])

        for _ in range(self.budget // self.pop_size):
            for i in range(self.pop_size):
                mutant = self.de_mutation(pop, i)
                trial = self.crossover(pop[i], mutant)
                f_trial = func(trial)
                
                # Accept with probability based on simulated annealing
                if f_trial < fit[i] or np.random.rand() < np.exp((fit[i] - f_trial) / self.temperature):
                    pop[i] = trial
                    fit[i] = f_trial

                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial

            self.temperature *= self.cooling_rate  # Cool down

        return self.f_opt, self.x_opt