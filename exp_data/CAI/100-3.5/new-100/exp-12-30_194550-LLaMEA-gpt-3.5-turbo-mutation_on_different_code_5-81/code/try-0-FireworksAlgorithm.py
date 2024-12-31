import numpy as np

class FireworksAlgorithm:
    def __init__(self, budget=10000, dim=10, sparks=5, max_sparks=50):
        self.budget = budget
        self.dim = dim
        self.sparks = sparks
        self.max_sparks = max_sparks
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        for i in range(self.budget):
            sparks_positions = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.sparks, self.dim))
            sparks_fitness = np.array([func(x) for x in sparks_positions])

            best_spark_idx = np.argmin(sparks_fitness)
            if sparks_fitness[best_spark_idx] < self.f_opt:
                self.f_opt = sparks_fitness[best_spark_idx]
                self.x_opt = sparks_positions[best_spark_idx]

            if self.sparks < self.max_sparks:
                selected_spark = sparks_positions[best_spark_idx]
                for j in range(self.sparks):
                    new_spark = selected_spark + np.random.normal(0, 1, self.dim)
                    new_fitness = func(new_spark)
                    if new_fitness < sparks_fitness[j]:
                        sparks_positions[j] = new_spark
                        sparks_fitness[j] = new_fitness
                        if new_fitness < self.f_opt:
                            self.f_opt = new_fitness
                            self.x_opt = new_spark

        return self.f_opt, self.x_opt