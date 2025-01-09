import numpy as np

class HybridFireworksAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.num_sparks = 5
        self.amplification = 0.1

    def __call__(self, func):
        def create_sparks(center, sigma):
            return center + np.random.normal(0, sigma, (self.num_sparks, self.dim))

        def differential_evolution(population, f_weight=0.5, f_cross=0.9):
            for i in range(self.population_size):
                a, b, c = np.random.choice(population, 3, replace=False)
                mutant = np.clip(a + f_weight * (b - c), -5.0, 5.0)
                cross_points = np.random.rand(self.dim) < f_cross
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                if func(trial) < func(population[i]):
                    population[i] = trial
            return population

        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        best_solution = population[np.argmin([func(x) for x in population])]
        
        for _ in range(self.budget):
            sparks = create_sparks(best_solution, self.amplification)
            sparks_fitness = [func(s) for s in sparks]
            best_spark = sparks[np.argmin(sparks_fitness)]
            if func(best_spark) < func(best_solution):
                best_solution = best_spark
            population = differential_evolution(population)
        
        return best_solution