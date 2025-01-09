import numpy as np

class AdaptiveFireworksAlgorithm:
    def __init__(self, budget, dim, explosion_amp=0.1, sparks_num=5):
        self.budget = budget
        self.dim = dim
        self.explosion_amp = explosion_amp
        self.sparks_num = sparks_num

    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, size=self.dim)
        best_fitness = func(best_solution)
        
        for _ in range(self.budget):
            for _ in range(self.sparks_num):
                spark = best_solution + np.random.normal(0, self.explosion_amp, size=self.dim)
                spark_fitness = func(spark)
                if spark_fitness < best_fitness:
                    best_solution = spark
                    best_fitness = spark_fitness
        
        return best_solution