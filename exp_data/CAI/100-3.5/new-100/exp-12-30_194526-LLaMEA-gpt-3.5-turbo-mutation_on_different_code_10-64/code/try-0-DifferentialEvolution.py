import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.8, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        pop_size = 10 * self.dim
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        population = np.random.uniform(bounds[0], bounds[1], size=(pop_size, self.dim))
        
        for i in range(self.budget):
            for j in range(pop_size):
                idxs = np.arange(pop_size)
                np.random.shuffle(idxs)
                a, b, c = population[np.random.choice(idxs[:3], 3, replace=False)]
                mask = np.random.rand(self.dim) < self.CR
                trial_vector = np.where(mask, a + self.F * (b - c), population[j])
                trial_vector = np.clip(trial_vector, bounds[0], bounds[1])

                f = func(trial_vector)
                if f < func(population[j]):
                    population[j] = trial_vector

                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = trial_vector
            
        return self.f_opt, self.x_opt