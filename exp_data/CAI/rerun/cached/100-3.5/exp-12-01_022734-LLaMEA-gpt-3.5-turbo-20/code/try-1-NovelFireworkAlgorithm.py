import numpy as np

class NovelFireworkAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def levy_flight(beta=1.5):
            sigma1 = (np.sqrt(np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (-1 / beta)
            sigma2 = 1
            u = np.random.normal(0, sigma1, self.dim)
            v = np.random.normal(0, sigma2, self.dim)
            step = u / (np.abs(v) ** (1 / beta))
            return step

        def init_firework():
            return np.random.uniform(-5.0, 5.0, self.dim)

        def explode(firework, sparks_num):
            sparks = np.zeros((sparks_num, self.dim))
            for i in range(sparks_num):
                sparks[i] = firework + levy_flight()
            return sparks

        population_size = 10
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_score = float('inf')

        for _ in range(self.budget):
            fireworks = [init_firework() for _ in range(population_size)]
            for firework in fireworks:
                sparks = explode(firework, population_size)
                for spark in sparks:
                    score = func(spark)
                    if score < best_score:
                        best_solution = spark
                        best_score = score
            population_size = int(1.1 * population_size)  # Dynamic population size

        return best_solution