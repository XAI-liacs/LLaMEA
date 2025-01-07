import numpy as np

class QuantumAntColonyOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_ants = 50
        self.pheromone = np.ones((self.num_ants, dim))
        self.best_path = None
        self.best_path_fitness = np.inf
        self.fitness_evaluations = 0
        self.memory = np.random.uniform(0.1, 0.5, self.num_ants)
        self.evaporation_rate = 0.2

    def quantum_noise(self):
        return np.random.uniform(-1, 1, size=self.dim)

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        lower_bound, upper_bound = bounds[0], bounds[1]

        while self.fitness_evaluations < self.budget:
            all_paths = []
            all_fitnesses = []

            for ant in range(self.num_ants):
                if self.fitness_evaluations >= self.budget:
                    break

                path = np.random.normal(self.pheromone[ant], scale=self.memory[ant])
                path = np.clip(path, lower_bound, upper_bound)

                fitness = func(path)
                self.fitness_evaluations += 1

                all_paths.append(path)
                all_fitnesses.append(fitness)

                if fitness < self.best_path_fitness:
                    self.best_path_fitness = fitness
                    self.best_path = path.copy()

            # Update pheromone trails based on fitness
            for ant in range(self.num_ants):
                if all_fitnesses[ant] < self.best_path_fitness:
                    self.pheromone[ant] = (1 - self.evaporation_rate) * self.pheromone[ant] + self.evaporation_rate * (all_paths[ant] - self.best_path)

                # Introduce quantum-based random exploration
                if np.random.rand() < 0.1:
                    self.pheromone[ant] += self.quantum_noise()

        return self.best_path