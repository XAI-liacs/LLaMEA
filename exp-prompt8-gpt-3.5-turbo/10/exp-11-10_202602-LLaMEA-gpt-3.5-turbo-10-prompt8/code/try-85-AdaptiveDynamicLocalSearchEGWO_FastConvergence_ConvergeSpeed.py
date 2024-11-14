import numpy as np

class AdaptiveDynamicLocalSearchEGWO_FastConvergence_ConvergeSpeed:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def update_position(position, best, a, c):
            return np.clip(position + a * (2 * np.random.rand(self.dim) - 1) * np.abs(c * best - position), -5.0, 5.0)

        def de_mutation(x, population, f):
            scaling_factor = 0.8 + 0.2 * np.random.rand()  # Introduce adaptive scaling factor
            a, b, c = population[np.random.choice(population.shape[0], 3, replace=False)]
            return np.clip(a + f * scaling_factor * (b - c), -5.0, 5.0)

        positions = np.random.uniform(-5.0, 5.0, (5, self.dim))
        fitness = np.array([func(p) for p in positions])
        best_idx = np.argmin(fitness)
        best_position = positions[best_idx]
        
        f = 0.5  # Initial mutation factor
        f_decay = 0.95  # Decay factor for the mutation factor

        fitness_improvement_rate = 0  # Initialize fitness improvement rate

        for _ in range(self.budget - 5):
            a = 2 - 2 * _ / (self.budget - 1)  # linearly decreasing a value
            for i in range(5):
                if i == best_idx:
                    continue
                c1 = 2 * np.random.rand(self.dim)
                c2 = 2 * np.random.rand(self.dim)
                c3 = 2 * np.random.rand(self.dim)
                if np.random.rand() > 0.5:  
                    positions[i] = update_position(positions[i], best_position, c1, c2)
                else:
                    positions[i] = de_mutation(positions[i], positions, f)

                if np.random.rand() < 0.3:  
                    positions[i] = update_position(positions[i], best_position, c1, c2)
                    
            new_fitness = np.array([func(p) for p in positions])
            new_best_idx = np.argmin(new_fitness)
            if new_fitness[new_best_idx] < fitness[best_idx]:
                fitness_improvement_rate = (fitness[best_idx] - new_fitness[new_best_idx]) / fitness[best_idx]
                fitness[best_idx] = new_fitness[new_best_idx]
                best_idx = new_best_idx
                best_position = positions[best_idx]

            if np.random.rand() < 0.1:  
                f = max(f * (1 + fitness_improvement_rate), 0.1)  # Adjust mutation factor based on fitness improvement rate

        return best_position