import numpy as np

class EnhancedFastFireworksAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        population_size = 10
        elitism_percentage = 0.2  # Preserve top 20% of individuals
        fireworks = np.random.uniform(-5.0, 5.0, size=(population_size, self.dim))
        best_firework = fireworks[np.argmin([func(firework) for firework in fireworks])
        for _ in range((self.budget // population_size) - 1):
            sparks = np.random.uniform(-0.1, 0.1, size=(population_size, self.dim))
            diversity_factor = np.mean(np.std(fireworks, axis=0)
            for i in range(population_size):
                for j in range(self.dim):
                    sparks[i][j] *= diversity_factor * np.abs(best_firework[j] - fireworks[i][j])
            fireworks += sparks
            combined_fireworks = np.vstack((fireworks, best_firework))
            indices = np.argsort([func(firework) for firework in combined_fireworks])[:int((1 - elitism_percentage) * population_size)]
            fireworks = combined_fireworks[indices]
            best_firework = fireworks[np.argmin([func(firework) for firework in fireworks])]
        return best_firework