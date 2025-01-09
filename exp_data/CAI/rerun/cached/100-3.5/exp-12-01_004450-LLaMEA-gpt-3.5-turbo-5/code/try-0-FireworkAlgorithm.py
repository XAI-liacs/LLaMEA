import numpy as np

class FireworkAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def generate_fireworks(n, dim):
            return np.random.uniform(-5.0, 5.0, size=(n, dim))

        def explode_fireworks(fireworks, n_explosions):
            explosions = np.repeat(fireworks, n_explosions, axis=0)
            spread = np.random.uniform(-1, 1, size=explosions.shape)
            return explosions + spread

        n_fireworks = 10
        n_explosions = 5
        fireworks = generate_fireworks(n_fireworks, self.dim)
        best_solution = None
        best_fitness = np.inf

        for _ in range(self.budget):
            fireworks = explode_fireworks(fireworks, n_explosions)
            fitness = np.array([func(firework) for firework in fireworks])
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fitness:
                best_solution = fireworks[min_idx]
                best_fitness = fitness[min_idx]

        return best_solution