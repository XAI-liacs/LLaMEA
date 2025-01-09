import numpy as np

class AdaptiveFireworksAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        pop_size = 10
        explosion_amp = 0.5
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)
        
        for _ in range(self.budget // pop_size):
            fireworks = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(pop_size)]
            fireworks_fitness = [func(fw) for fw in fireworks]
            
            for i in range(pop_size):
                for j in range(pop_size):
                    if fireworks_fitness[j] < fireworks_fitness[i]:
                        dist = np.linalg.norm(fireworks[j] - fireworks[i])
                        if dist != 0:
                            fireworks[i] += explosion_amp * (fireworks[j] - fireworks[i]) / dist
            
            best_firework = fireworks[np.argmin(fireworks_fitness)]
            best_firework_fitness = func(best_firework)
            
            if best_firework_fitness < best_fitness:
                best_solution = best_firework
                best_fitness = best_firework_fitness
            
            explosion_amp *= 0.9  # Decrease explosion amplitude over time
        
        return best_solution