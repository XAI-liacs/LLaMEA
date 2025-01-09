import numpy as np

class HybridDESA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.bounds = (-5.0, 5.0)
        self.temperature = 1.0
        self.temperature_decay = 0.99

    def __call__(self, func):
        np.random.seed(42)  # For reproducibility
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = a + np.random.rand(self.dim) * (b - c)
                mutant = np.clip(mutant, self.bounds[0], self.bounds[1])

                # Crossover
                cross_points = np.random.rand(self.dim) < 0.9
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, population[i])
                trial_fitness = func(trial)
                evaluations += 1

                # Acceptance based on Simulated Annealing
                delta_f = trial_fitness - fitness[i]
                if delta_f < 0 or np.exp(-delta_f / self.temperature) > np.random.rand():
                    population[i] = trial
                    fitness[i] = trial_fitness

                if evaluations >= self.budget:
                    break

            self.temperature *= self.temperature_decay

        return population[np.argmin(fitness)]

# Example usage:
# optimizer = HybridDESA(budget=1000, dim=10)
# best_solution = optimizer(your_black_box_function_here)