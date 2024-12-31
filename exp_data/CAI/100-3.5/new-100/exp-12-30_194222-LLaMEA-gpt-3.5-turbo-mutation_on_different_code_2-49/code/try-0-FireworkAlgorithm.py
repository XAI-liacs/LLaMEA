import numpy as np

class FireworkAlgorithm:
    def __init__(self, budget=10000, dim=10, n_fireworks=10, n_sparks=5):
        self.budget = budget
        self.dim = dim
        self.n_fireworks = n_fireworks
        self.n_sparks = n_sparks
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        fireworks = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.n_fireworks, self.dim))
        
        for _ in range(self.budget):
            for fw in fireworks:
                sparks = np.random.uniform(fw - 0.1, fw + 0.1, size=(self.n_sparks, self.dim))
                for spark in sparks:
                    f = func(spark)
                    if f < self.f_opt:
                        self.f_opt = f
                        self.x_opt = spark
                        
        return self.f_opt, self.x_opt