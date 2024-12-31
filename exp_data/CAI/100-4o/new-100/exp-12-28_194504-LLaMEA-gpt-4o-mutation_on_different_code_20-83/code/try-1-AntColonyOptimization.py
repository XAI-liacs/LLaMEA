import numpy as np

class AntColonyOptimization:
    def __init__(self, budget=10000, dim=10, num_ants=50, evaporation_rate=0.1, alpha=1.0, beta=2.0):
        self.budget = budget
        self.dim = dim
        self.num_ants = num_ants
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta
        self.f_opt = np.Inf
        self.x_opt = None
        self.pheromone = np.ones((num_ants, dim))
        self.heuristic = np.random.uniform(-5, 5, (num_ants, dim))

    def __call__(self, func):
        evaluations = 0

        while evaluations < self.budget:
            ants_positions = np.zeros((self.num_ants, self.dim))

            for i in range(self.num_ants):
                for j in range(self.dim):
                    probabilities = (self.pheromone[i, j] ** self.alpha) * (self.heuristic[i, j] ** self.beta)
                    probabilities /= probabilities.sum()
                    ants_positions[i, j] = np.random.choice(self.heuristic[i], p=probabilities)
            
            ants_positions += np.random.normal(0, 0.1, ants_positions.shape)  # Add Gaussian noise

            for i in range(self.num_ants):
                x = np.clip(ants_positions[i], -5, 5)
                f = func(x)
                evaluations += 1

                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = x

            self.pheromone *= (1 - self.evaporation_rate)

            for i in range(self.num_ants):
                f_current = func(ants_positions[i])
                if f_current < self.f_opt:
                    self.pheromone[i] += 1 / (1 + f_current)

        return self.f_opt, self.x_opt