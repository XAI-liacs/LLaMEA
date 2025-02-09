import numpy as np

class OppositionBasedBatAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.population = None
        self.velocities = None
        self.lb = None
        self.ub = None
        self.frequencies = None
        self.loudness = 0.5
        self.pulse_rate = 0.5
        self.best = None
        self.best_score = np.inf

    def initialize_population(self, lb, ub, size):
        self.population = lb + (ub - lb) * np.random.rand(size, self.dim)
        self.velocities = np.zeros((size, self.dim))
        self.frequencies = np.zeros(size)

    def opposition_based_learning(self):
        return self.lb + self.ub - self.population

    def periodic_constraint(self, position):
        period = (self.ub - self.lb) / self.dim
        period_position = self.lb + (np.round((position - self.lb) / period) * period)
        return np.clip(period_position, self.lb, self.ub)

    def bat_algorithm(self, func):
        ob_population = self.opposition_based_learning()
        ob_scores = np.array([func(x) for x in ob_population])
        ob_best_idx = np.argmin(ob_scores)

        for _ in range(self.budget - self.population_size):
            for i in range(self.population_size):
                self.frequencies[i] = np.random.rand()
                new_velocity = self.velocities[i] + (self.population[i] - self.best) * self.frequencies[i]
                new_solution = self.population[i] + new_velocity
                new_solution = self.periodic_constraint(new_solution)

                if np.random.rand() > self.pulse_rate:
                    new_solution = self.best + 0.001 * np.random.randn(self.dim)

                new_score = func(new_solution)

                if (new_score < self.best_score) and (np.random.rand() < self.loudness):
                    self.population[i] = new_solution
                    self.velocities[i] = new_velocity
                    if new_score < self.best_score:
                        self.best = new_solution
                        self.best_score = new_score

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(self.lb, self.ub, self.population_size)
        self.best = self.population[np.argmin([func(x) for x in self.population])]
        self.best_score = func(self.best)
        self.bat_algorithm(func)
        return self.best