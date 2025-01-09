import numpy as np

class DragonflyAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.step_size = 0.1
        self.attract_repel_factor = 0.1

    def __call__(self, func):
        def levy_flight():
            beta = 1.5
            sigma = (np.math.gamma(1 + beta) * np.math.sin(np.pi * beta / 2) / (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
            u = np.random.normal(0, sigma, self.dim)
            v = np.random.normal(0, 1, self.dim)
            step = u / (np.abs(v) ** (1 / beta))
            return step

        def move_towards_target(dragonfly, target):
            direction = target - dragonfly
            step = self.step_size * levy_flight()
            return dragonfly + step * direction

        def attract_repel(dragonflies, fitnesses):
            best_idx = np.argmin(fitnesses)
            worst_idx = np.argmax(fitnesses)
            attractor = np.mean(dragonflies, axis=0)
            repeller = dragonflies[worst_idx]
            return dragonflies + self.attract_repel_factor * (attractor - dragonflies) - self.attract_repel_factor * (repeller - dragonflies)

        dragonflies = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        fitnesses = np.array([func(x) for x in dragonflies])

        for _ in range(self.budget - self.population_size):
            dragonflies = np.array([move_towards_target(dragonflies[i], dragonflies[np.argmin(fitnesses)]) for i in range(self.population_size)])
            dragonflies = attract_repel(dragonflies, fitnesses)
            fitnesses = np.array([func(x) for x in dragonflies])

        best_idx = np.argmin(fitnesses)
        return dragonflies[best_idx]