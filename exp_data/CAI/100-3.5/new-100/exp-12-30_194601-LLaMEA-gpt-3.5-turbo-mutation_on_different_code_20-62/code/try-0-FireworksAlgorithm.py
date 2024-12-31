import numpy as np

class FireworksAlgorithm:
    def __init__(self, budget=10000, dim=10, n_fireworks=10, n_sparks=5, alpha=0.5, beta=2.0):
        self.budget = budget
        self.dim = dim
        self.n_fireworks = n_fireworks
        self.n_sparks = n_sparks
        self.alpha = alpha
        self.beta = beta
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        fireworks = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.n_fireworks, self.dim))
        brightness = [func(fw) for fw in fireworks]
        
        for _ in range(self.budget):
            for i in range(self.n_fireworks):
                for _ in range(self.n_sparks):
                    spark = fireworks[i] + self.alpha * np.random.uniform(-1, 1, size=self.dim) + self.beta * np.random.normal(size=self.dim)
                    spark_fitness = func(spark)
                    
                    if spark_fitness < brightness[i]:
                        brightness[i] = spark_fitness
                        fireworks[i] = spark
                        
            best_idx = np.argmin(brightness)
            if brightness[best_idx] < self.f_opt:
                self.f_opt = brightness[best_idx]
                self.x_opt = fireworks[best_idx]
                
        return self.f_opt, self.x_opt