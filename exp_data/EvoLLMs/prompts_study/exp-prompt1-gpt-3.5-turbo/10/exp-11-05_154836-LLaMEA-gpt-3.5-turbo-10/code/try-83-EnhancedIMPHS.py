import numpy as np

class EnhancedIMPHS(IMPHS):
    def exploit_phase(self, population, num_iterations=5):
        for _ in range(num_iterations):
            best_idx = np.argmin(self.evaluate_population(population))
            best_individual = population[best_idx]
            new_population = population + np.random.normal(0, 0.1, population.shape) * np.random.uniform(0.1, 0.5, (1, self.dim))
            new_population[best_idx] = best_individual
            new_population = np.clip(new_population, self.lower_bound, self.upper_bound)
            population = new_population
        return population