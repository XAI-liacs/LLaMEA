import numpy as np

class FireworksAlgorithm:
    def __init__(self, budget=10000, dim=10, num_sparks=5, num_parents=3, mutation_rate=0.1, levy_alpha=1.5):
        self.budget = budget
        self.dim = dim
        self.num_sparks = num_sparks
        self.num_parents = num_parents
        self.mutation_rate = mutation_rate
        self.levy_alpha = levy_alpha
        self.f_opt = np.Inf
        self.x_opt = None

    def levy_flight(self, scale=0.1):
        sigma = (np.abs(np.random.normal(0, 1)) / (np.abs(np.random.normal(0, 1))) ** (1 / self.levy_alpha)) ** (1 / self.levy_alpha)
        step = np.random.normal(0, scale)
        return step * sigma

    def __call__(self, func):
        sparks = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.num_sparks, self.dim))
        
        for i in range(self.budget):
            parents = sparks[np.argsort([func(x) for x in sparks])[:self.num_parents]]
            
            for j in range(self.num_sparks):
                mutation = np.random.normal(0, self.mutation_rate, size=self.dim)
                sparks[j] = parents[np.random.randint(0, self.num_parents)] + mutation + self.levy_flight()
            
            best_spark_index = np.argmin([func(x) for x in sparks])
            best_spark = sparks[best_spark_index]
            if func(best_spark) < self.f_opt:
                self.f_opt = func(best_spark)
                self.x_opt = best_spark
            
        return self.f_opt, self.x_opt