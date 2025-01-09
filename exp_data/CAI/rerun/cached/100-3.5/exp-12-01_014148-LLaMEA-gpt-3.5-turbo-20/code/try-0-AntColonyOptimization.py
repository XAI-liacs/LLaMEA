import numpy as np

class AntColonyOptimization:
    def __init__(self, budget, dim, num_ants=10, evaporation_rate=0.95, alpha=1.0, beta=2.0):
        self.budget = budget
        self.dim = dim
        self.num_ants = num_ants
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta

    def __call__(self, func):
        def initialize_ants():
            return np.random.uniform(-5.0, 5.0, size=(self.num_ants, self.dim))

        def evaluate_ant(ant):
            return func(ant)

        def update_pheromones(pheromones, best_ant):
            pheromones *= self.evaporation_rate
            pheromones += 1.0 / (1.0 + evaluate_ant(best_ant))

        best_ant = None
        pheromones = np.ones(self.dim)

        for _ in range(self.budget):
            ants = initialize_ants()
            scores = np.array([evaluate_ant(ant) for ant in ants])
            best_ant = ants[np.argmin(scores)]
            update_pheromones(pheromones, best_ant)

        return best_ant