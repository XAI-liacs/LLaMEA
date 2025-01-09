import numpy as np

class FireworksAlgorithm:
    def __init__(self, budget=10000, dim=10, num_sparks=5):
        self.budget = budget
        self.dim = dim
        self.num_sparks = num_sparks
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        for i in range(self.budget):
            sparks = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.num_sparks, self.dim))
            fireworks = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.num_sparks, self.dim))
            
            for firework in fireworks:
                f = func(firework)
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = firework
            
            best_spark = sparks[np.argmin([func(spark) for spark in sparks])]
            for j, firework in enumerate(fireworks):
                new_firework = firework + np.random.uniform(-1, 1, size=self.dim) * (best_spark - firework)
                f_new = func(new_firework)
                if f_new < func(firework):
                    fireworks[j] = new_firework
            
        return self.f_opt, self.x_opt