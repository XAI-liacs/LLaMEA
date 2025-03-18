import numpy as np

class AntColonyGradientBasedOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_ants = 10 * self.dim
        self.pheromone = 0.1  # Initial pheromone strength
        self.evaporation_rate = 0.1  # Pheromone evaporation rate
        self.best_position = None
        self.best_value = np.inf

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        positions = np.random.uniform(lb, ub, (self.num_ants, self.dim))
        evaluations = 0

        while evaluations < self.budget:
            fitness_values = np.array([func(pos) for pos in positions])
            evaluations += self.num_ants

            if np.min(fitness_values) < self.best_value:
                self.best_value = np.min(fitness_values)
                self.best_position = positions[np.argmin(fitness_values)]

            # Apply pheromone update rule
            sorted_indices = np.argsort(fitness_values)
            for rank, idx in enumerate(sorted_indices):
                reinforcement = (self.num_ants - rank) / self.num_ants
                for d in range(self.dim):
                    delta = np.random.normal(0, self.pheromone)
                    positions[idx][d] += reinforcement * delta
                    positions[idx][d] = np.clip(positions[idx][d], lb[d], ub[d])

            # Evaporate old pheromone
            self.pheromone *= (1 - self.evaporation_rate)

            # Adaptive gradient descent step
            for i in range(self.num_ants):
                gradient = (positions[i] - self.best_position) / np.linalg.norm(positions[i] - self.best_position)
                step_size = self.pheromone * np.random.rand()
                positions[i] = np.clip(positions[i] - step_size * gradient, lb, ub)

        return self.best_position, self.best_value